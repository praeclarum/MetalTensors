using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Foundation;
using Metal;
using MetalPerformanceShaders;
using MetalTensors.Layers;
using MetalTensors.Tensors;

namespace MetalTensors
{
    public class TrainingGraph : Graph
    {
        readonly (ConvDataSource Weights, bool Trainable)[] convWeights;

        readonly bool[] intermediateIsLoss;

        public TrainingGraph (string label, Tensor[] inputs, Tensor[] outputs, Tensor[] losses, Dictionary<Layer, bool> trainable, IMTLDevice device)
            : base (label, CreateTrainingGraph (label, losses, trainable, device, out var cweights), inputs, outputs, device)
        {
            convWeights = cweights;
            intermediateIsLoss = new bool[intermediateHandles.Length];
            for (var i = 0; i < intermediateHandles.Length; i++) {
                var handle = intermediateHandles[i];
                if (losses.FirstOrDefault (x => x.Label == handle.Label) != null) {
                    intermediateIsLoss[i] = true;
                }
            }
        }

        static MPSNNGraph CreateTrainingGraph (string label, Tensor[] losses, Dictionary<Layer, bool> trainable, IMTLDevice device, out (ConvDataSource Weights, bool Trainable)[] cweights)
        {
            if (losses.Length < 1) {
                throw new ArgumentException ("Loss is required in order to train", nameof(losses));
            }
            //stopwatch.Start ();

            //
            // Build the training graph
            //
            var context = new MetalImageNodeContext (label, false, device);
            var trainingOutput = losses.Length == 1 ? losses[0] : Tensor.Sum (losses);
            var thisImageNode = trainingOutput.GetImageNode (context);

            var initialGrad = new MPSNNInitialGradientNode (thisImageNode);
            var convWeightsL = new List<(ConvDataSource, bool)> ();

            var trainingGraphTermini = initialGrad.GetTrainingGraph (null, (gradientNode, inferenceNode, inferenceSource, gradientSource) => {
                //Console.WriteLine ($"gradientNode={gradientNode}, inferenceNode={inferenceNode}, inferenceSource={inferenceSource}, gradientSource={gradientSource}");
                gradientNode.ResultImage.Format = MPSImageFeatureChannelFormat.Float32;
                if (inferenceNode.ResultImage.MPSHandle is LayerHandle lh &&
                         lh.Layer.GetMetalConvDataSource (device) is ConvDataSource cw) {
                    if (!trainable.ContainsKey (lh.Layer)) {
                        throw new Exception ($"Cannot tell if {lh.Layer} is trainable");
                    }
                    var train = trainable[lh.Layer];
                    convWeightsL.Add ((cw, train));
                    
                    //Console.WriteLine (lh);
                }                
            });

            cweights = convWeightsL.ToArray ();

            var trainingGraphTerminiImageNodes = trainingGraphTermini.Select (x => x.ResultImage).ToArray ();
            var resultsNeeded = trainingGraphTerminiImageNodes.Select (x => true).ToArray ();

            //
            // Export all losses
            //
            foreach (var t in losses) {
                ExportTensor (t, context);
            }

            //
            // Create the graph
            //
            var trainingGraph = MPSNNGraph.Create (device, trainingGraphTerminiImageNodes, resultsNeeded);
            trainingGraph.Format = MPSImageFeatureChannelFormat.Float32;

            return trainingGraph;
        }

        public TrainingHistory Fit (DataSet dataSet, Optimizer optimizer, int batchSize, int numBatches, int validateInterval, EvaluationGraph? evalGraph)
        {
            //Tensor[][] inputsBatch, Tensor[][] outputsBatch
            if (validateInterval <= 0)
                throw new ArgumentException ($"Invalidate validation interval ({validateInterval}) specified.");

            using var pool = new NSAutoreleasePool ();

            MetalGraph.ReloadFromDataSources ();

            //
            // Set the learning rate
            //
            foreach (var c in convWeights) {
                c.Weights.SetOptimizationOptions (c.Trainable, optimizer.LearningRate);
            }

            //
            // Init history
            //
            var h = new List<TrainingHistory.BatchHistory> ();
            void AddHistory (TrainingHistory.BatchHistory bh)
            {
                lock (h) {
                    h.Add (bh);
                }
            }

            //
            // Train
            //
            using var queue = Device.CreateCommandQueue ();
            if (queue == null)
                throw new Exception ("Failed to create command queue");

            var semaphore = new Semaphore (2, 2);

            MPSCommandBuffer? lcb = null;
            for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {
                var (inputs, outputs) = dataSet.GetBatch (batchIndex * batchSize, batchSize, queue.Device);
                lcb = EncodeTrainingBatch (inputs, outputs, AddHistory, semaphore, queue);

                if (evalGraph != null && ((batchIndex + 1) % validateInterval == 0)) {
                    lcb?.WaitUntilCompleted ();
                    lcb = null;
                    var evalHistory = evalGraph.Evaluate (dataSet, batchSize, 1, semaphore, queue);
                    //Console.WriteLine (evalHistory);
                }
            }
            lcb?.WaitUntilCompleted ();

            return new TrainingHistory (h);
        }

        public TrainingHistory.BatchHistory Fit (Tensor[][] inputsBatch, Tensor[][] outputsBatch, Optimizer optimizer)
        {
            var batchSize = inputsBatch.Length;

            using var pool = new NSAutoreleasePool ();

            MetalGraph.ReloadFromDataSources ();

            //
            // Set the learning rate
            //
            foreach (var c in convWeights) {
                c.Weights.SetOptimizationOptions (c.Trainable, optimizer.LearningRate);
            }

            //
            // Init history
            //
            var h = new List<TrainingHistory.BatchHistory> ();
            void AddHistory (TrainingHistory.BatchHistory bh)
            {
                lock (h) {
                    h.Add (bh);
                }
            }

            //
            // Train
            //
            using var queue = Device.CreateCommandQueue ();
            if (queue == null)
                throw new Exception ("Failed to create command queue");

            using var semaphore = new Semaphore (2, 2);

            MPSCommandBuffer lcb = EncodeTrainingBatch (inputsBatch, outputsBatch, AddHistory, semaphore, queue);

            lcb.WaitUntilCompleted ();

            return h[0];
        }

        MPSCommandBuffer EncodeTrainingBatch (Tensor[][] inputs, Tensor[][] outputs, Action<TrainingHistory.BatchHistory> recordHistory, Semaphore semaphore, IMTLCommandQueue queue)
        {
            if (inputs.Length < 1)
                throw new ArgumentException ($"At least one input is needed in a batch");

            //
            // This pool is necessary for Metal to clean up its objects
            //
            using var pool = new NSAutoreleasePool ();

            //
            // Load data
            //
            var (batch, temporaryBatchImages) = GetSourceImages (inputs, outputs);

            //Console.WriteLine ($"BATCH BYTE SIZE {batchSize*(2+1)*4:#,0}");

            //
            // Wait for the last command to finish
            //
            semaphore.WaitOne ();

            //Console.WriteLine ($"{stopwatch.Elapsed} START BATCH {batchIndex} (thread {Thread.CurrentThread.ManagedThreadId})");

            // No using because it is returned
            var commandBuffer = MPSCommandBuffer.Create (queue);

            //
            // Encode the graph
            //
            using var intermediateImagesMA = new NSMutableArray<NSArray<MPSImage>> ();
            var destinationStates = new NSMutableArray<NSArray<MPSState>> ();
            var returnBatch = MetalGraph.EncodeBatch (commandBuffer, batch, null, intermediateImagesMA, null);
            var intermediateImages = intermediateImagesMA.ToArray ();

            //
            // Synchronize needed images
            //
            for (var i = 0; i < intermediateHandles.Length; i++) {
                if (i < intermediateImages.Length) {
                    if (intermediateIsLoss[i]) {
                        MPSImageBatch.Synchronize (intermediateImages[i], commandBuffer);
                    }
                }
            }

            //
            // Setup the completed callback
            //
            commandBuffer.AddCompletedHandler (cmdBuf => {

                //Console.WriteLine ($"{stopwatch.Elapsed} END BATCH {batchIndex} (thread {Thread.CurrentThread.ManagedThreadId})");
                semaphore.Release ();

                if (cmdBuf.Error != null) {
                    Console.WriteLine ($"{Label}: Command Buffer Error: {cmdBuf.Error.Description}");
                }

                //
                // Don't need the returnBatch
                //
                if (returnBatch != null) {
                    foreach (var i in returnBatch) {
                        i.Dispose ();
                    }
                    returnBatch.Dispose ();
                    returnBatch = null;
                }

                //
                // Record the loss history
                //
                //Console.WriteLine ($"{intermediateImages.Length} ims");                
                var loss = intermediateImages.Length > 0 ?
                    intermediateImages[0].Select (x => new MPSImageTensor (x)).ToArray () :
                    Array.Empty<Tensor> ();
                var losses = new Dictionary<string, float> ();
                for (var i = 0; i < intermediateHandles.Length; i++) {
                    if (i < intermediateImages.Length) {
                        if (intermediateIsLoss[i]) {
                            losses[intermediateHandles[i].Label] = intermediateImages[i].ReduceMeanValue ();
                        }
                    }
                }
                intermediateImages.Dispose ();

                //
                // Broadcast the results to whomever is listening
                //
                var h = new TrainingHistory.BatchHistory (Array.Empty<Tensor> (), losses, new Dictionary<string, Tensor[]>(), temporaryBatchImages, cmdBuf.Device);
                recordHistory (h);
            });

            //
            // Run the batch
            //
            commandBuffer.Commit ();

            return commandBuffer;
        }
    }
}
