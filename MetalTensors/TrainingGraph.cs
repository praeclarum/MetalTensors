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
        readonly (IWeightsDataSource Weights, bool Trainable)[] weightDataSources;

        readonly bool[] intermediateIsLoss;

        readonly Optimizer optimizer;

        public TrainingGraph (string label, Tensor[] inputs, Tensor[] outputs, Tensor[] losses, Dictionary<Layer, bool> trainable, Optimizer optimizer, IMTLCommandQueue queue, Semaphore semaphore)
            : base (label, CreateTrainingGraph (label, losses, trainable, queue.Device, out var cweights), inputs, outputs, queue, semaphore)
        {
            weightDataSources = cweights;
            intermediateIsLoss = new bool[intermediateHandles.Length];
            this.optimizer = optimizer;
            for (var i = 0; i < intermediateHandles.Length; i++) {
                var handle = intermediateHandles[i];
                if (losses.FirstOrDefault (x => x.Label == handle.Label) != null) {
                    intermediateIsLoss[i] = true;
                }
            }
        }

        static MPSNNGraph CreateTrainingGraph (string label, Tensor[] losses, Dictionary<Layer, bool> trainable, IMTLDevice device, out (IWeightsDataSource Weights, bool Trainable)[] weightDataSources)
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
            var weightsL = new List<(IWeightsDataSource, bool)> ();

            var trainingGraphTermini = initialGrad.GetTrainingGraph (null, (gradientNode, inferenceNode, inferenceSource, gradientSource) => {
                //Console.WriteLine ($"gradientNode={gradientNode}, inferenceNode={inferenceNode}, inferenceSource={inferenceSource}, gradientSource={gradientSource}");
                gradientNode.ResultImage.Format = MPSImageFeatureChannelFormat.Float32;
                if (inferenceNode.ResultImage.MPSHandle is LayerHandle lh &&
                         lh.Layer is WeightsLayer wl &&
                         wl.TryGetDataSource (device) is IWeightsDataSource cw) {
                    if (!trainable.ContainsKey (lh.Layer)) {
                        throw new Exception ($"Cannot tell if {lh.Layer} is trainable");
                    }
                    var train = trainable[lh.Layer];
                    weightsL.Add ((cw, train));                    
                    //Console.WriteLine (lh);
                }
            });

            weightDataSources = weightsL.ToArray ();

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

        public TrainingHistory Fit (DataSet dataSet, int batchSize, int numBatches, Action<TrainingHistory.BatchHistory>? callback)
        {
            //Tensor[][] inputsBatch, Tensor[][] outputsBatch
            //if (validateInterval <= 0)
            //    throw new ArgumentException ($"Invalidate validation interval ({validateInterval}) specified.");

            using var pool = new NSAutoreleasePool ();

            MetalGraph.ReloadFromDataSources ();

            //
            // Set the learning rate
            //
            foreach (var c in weightDataSources) {
                c.Weights.SetOptimizationOptions (c.Trainable, optimizer);
            }

            //
            // Init history
            //
            var h = new List<TrainingHistory.BatchHistory> ();
            var continueTraining = true;
            void AddHistory (TrainingHistory.BatchHistory bh)
            {
                lock (h) {
                    h.Add (bh);
                }
                callback?.Invoke (bh);
                continueTraining = bh.ContinueTraining;
            }

            MPSCommandBuffer? lcb = null;
            for (int batchIndex = 0; continueTraining && batchIndex < numBatches; batchIndex++) {
                var (inputs, outputs) = dataSet.GetBatch (batchIndex * batchSize, batchSize, Device);
                lcb = EncodeTrainingBatch (inputs, outputs, AddHistory);

                //if (evalGraph != null && ((batchIndex + 1) % validateInterval == 0)) {
                //    lcb?.WaitUntilCompleted ();
                //    lcb = null;
                //    var evalHistory = evalGraph.Evaluate (dataSet, batchSize, 1, semaphore, queue);
                //    //Console.WriteLine (evalHistory);
                //}
            }
            lcb?.WaitUntilCompleted ();

            return new TrainingHistory (h);
        }

        public TrainingHistory.BatchHistory Fit (Tensor[][] inputsBatch, Tensor[][] outputsBatch)
        {
            var batchSize = inputsBatch.Length;

            using var pool = new NSAutoreleasePool ();

            MetalGraph.ReloadFromDataSources ();

            //
            // Set the learning rate
            //
            foreach (var c in weightDataSources) {
                c.Weights.SetOptimizationOptions (c.Trainable, optimizer);
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
            MPSCommandBuffer lcb = EncodeTrainingBatch (inputsBatch, outputsBatch, AddHistory);

            lcb.WaitUntilCompleted ();

            return h[0];
        }

        

        MPSCommandBuffer EncodeTrainingBatch (Tensor[][] inputs, Tensor[][] outputs, Action<TrainingHistory.BatchHistory> recordHistory)
        {
            if (inputs.Length < 1)
                throw new ArgumentException ($"At least one input is needed in a batch");

            //
            // This pool is necessary for Metal to clean up its objects
            //
            using var pool = new NSAutoreleasePool ();

            //
            // Wait for the last command to finish
            //
            modelSemaphore.WaitOne ();

            var commandBuffer = MPSCommandBuffer.Create (modelQueue);
            commandBuffer.Label = $"{Label} {nextCommandId}";
            Interlocked.Increment (ref nextCommandId);

            //
            // Load data
            //
            var batchSize = inputs.Length;
            //var (batch, temporaryBatchImages) = GetSourceImages (inputs, outputs);

            var batchSourceImages = RentSourceImages (batchSize);
            //var cbatch = CopySourceImages (inputs, outputs, batchSourceImages, queue);
            var cbatch = EncodeSourceImages (inputs, outputs, batchSourceImages, commandBuffer);

            //Console.WriteLine ($"BATCH BYTE SIZE {batchSize*(2+1)*4:#,0}");


            //Console.WriteLine ($"{stopwatch.Elapsed} START BATCH {batchIndex} (thread {Thread.CurrentThread.ManagedThreadId})");

            // No using because it is returned

            //
            // Encode the graph
            //
            using var intermediateImagesMA = new NSMutableArray<NSArray<MPSImage>> ();
            var destinationStates = new NSMutableArray<NSArray<MPSState>> ();
            var returnBatch = MetalGraph.EncodeBatch (commandBuffer, cbatch, null, intermediateImagesMA, null);
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

                //
                // Keep the source images for the next batch
                //
                ReturnSourceImages (batchSourceImages);

                //Console.WriteLine ($"{stopwatch.Elapsed} END BATCH {batchIndex} (thread {Thread.CurrentThread.ManagedThreadId})");
                modelSemaphore.Release ();

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
                var losses = new Dictionary<string, float> ();
                for (var i = 0; i < intermediateHandles.Length; i++) {
                    if (i < intermediateImages.Length) {
                        if (intermediateIsLoss[i]) {
                            var image = intermediateImages[i];
                            losses[intermediateHandles[i].Label] = image.ReduceMeanValueAndDispose ();
                        }
                        else {
                            intermediateImages[i].Dispose ();
                        }
                    }
                }

                //
                // Broadcast the results to whomever is listening
                //
                var h = new TrainingHistory.BatchHistory (Array.Empty<Tensor> (), losses, new Dictionary<string, Tensor[]>(), cmdBuf.Device);
                recordHistory (h);
                cmdBuf.Dispose ();
            });

            //
            // Run the batch
            //
            commandBuffer.Commit ();

            return commandBuffer;
        }
    }
}
