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
    class TrainingGraph
    {
        readonly IMTLDevice device;
        readonly MPSNNGraph trainingGraph;
        readonly TensorHandle[] sourceHandles;
        readonly LayerHandle[] intermediateHandles;
        readonly (ConvWeights Weights, bool Trainable)[] convWeights;
        readonly Dictionary<Layer, bool> trainable;

        public TrainingGraph (Tensor output, Dictionary<Layer, bool> trainable, IMTLDevice device)
        {
            this.device = device;
            this.trainable = trainable;
            //stopwatch.Start ();

            //
            // Build the training graph
            //
            var thisImageNode = output.GetMetalImageNode (true, device);

            var initialGrad = new MPSNNInitialGradientNode (thisImageNode);
            var lossNodesIndex = new Dictionary<IntPtr, MPSNNForwardLossNode> ();
            var convWeightsL = new List<(ConvWeights, bool)> ();
            var trainingGraphTermini = initialGrad.GetTrainingGraph (null, (gradientNode, inferenceNode, inferenceSource, gradientSource) => {
                //Console.WriteLine ($"gradientNode={gradientNode}, inferenceNode={inferenceNode}, inferenceSource={inferenceSource}, gradientSource={gradientSource}");
                gradientNode.ResultImage.Format = MPSImageFeatureChannelFormat.Float32;
                if (inferenceNode is MPSNNForwardLossNode ln) {
                    lossNodesIndex[ln.Handle] = ln;
                }
                else if (inferenceNode.ResultImage.MPSHandle is LayerHandle lh &&
                         lh.Layer.GetMetalConvDataSource (device) is ConvWeights cw) {
                    convWeightsL.Add ((cw, trainable[lh.Layer]));
                    //Console.WriteLine (lh);
                }                
            });

            convWeights = convWeightsL.ToArray ();

            var lossNodes = lossNodesIndex.Values.ToArray ();
            if (lossNodes.Length < 1) {
                throw new InvalidOperationException ("Loss is required in order to train");
            }

            var trainingGraphTerminiImageNodes = trainingGraphTermini.Select (x => x.ResultImage).ToArray ();
            var resultsNeeded = trainingGraphTerminiImageNodes.Select (x => true).ToArray ();

            trainingGraph = MPSNNGraph.Create (device, trainingGraphTerminiImageNodes, resultsNeeded);
            trainingGraph.Format = MPSImageFeatureChannelFormat.Float32;

            sourceHandles = trainingGraph.SourceImageHandles.Select (x => (TensorHandle)x).ToArray ();
            //var resultStateHandles = trainingGraph.ResultStateHandles;
            intermediateHandles = trainingGraph.IntermediateImageHandles.Select (x => (LayerHandle)x).ToArray ();

            //Console.WriteLine (intermediateHandles);
            //Console.WriteLine (trainingGraph.DebugDescription);
        }

        public TrainingHistory Train (Func<TensorHandle[], IEnumerable<Tensor>> trainingData, float learningRate, int batchSize, int numBatches)
        {
            //
            // Set the learning rate
            //
            foreach (var c in convWeights) {
                c.Weights.SetOptimizationOptions (c.Trainable, learningRate);
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
            var stopwatch = new Stopwatch ();
            stopwatch.Restart ();
            using var q = device.CreateCommandQueue ();

            var semaphore = new Semaphore (2, 2);

            MPSCommandBuffer? lcb = null;
            for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {
                lcb = TrainBatch (batchIndex, trainingData, batchSize, AddHistory, stopwatch, semaphore, q);
            }
            if (lcb != null) {
                lcb.WaitUntilCompleted ();
            }

            return new TrainingHistory (h);
        }

        MPSCommandBuffer TrainBatch (int batchIndex, Func<TensorHandle[], IEnumerable<Tensor>> trainingData, int batchSize, Action<TrainingHistory.BatchHistory> recordHistory, Stopwatch stopwatch, Semaphore semaphore, IMTLCommandQueue queue)
        {
            //
            // This pool is necessary for Metal to clean up its objects
            //
            using var pool = new NSAutoreleasePool ();

            semaphore.WaitOne ();
            //Console.WriteLine ($"{stopwatch.Elapsed} START BATCH {batchIndex} (thread {Thread.CurrentThread.ManagedThreadId})");

            //
            // Load data
            //
            NSArray<MPSImage>[] batch;
            MPSImage[] temporaryBatchImages;
            try {
                (batch, temporaryBatchImages) = GetBatch (trainingData, batchSize);
            }
            catch {
                semaphore.Release ();
                throw;
            }

            //Console.WriteLine ($"BATCH BYTE SIZE {batchSize*(2+1)*4:#,0}");

            var commandBuffer = MPSCommandBuffer.Create (queue);

            //
            // Encode the graph
            //
            var intermediateImagesMA = new NSMutableArray<NSArray<MPSImage>> ();
            var destinationStates = new NSMutableArray<NSArray<MPSState>> ();
            NSArray<MPSImage>? returnBatch = trainingGraph.EncodeBatch (commandBuffer, batch, System.Array.Empty<NSArray<MPSState>> (), intermediateImagesMA, null);
            var intermediateImages = intermediateImagesMA.ToArray ();

            //
            // Synchronize needed images
            //
            if (returnBatch != null) {
                MPSImageBatch.Synchronize (returnBatch, commandBuffer);
            }
            foreach (var imBatch in intermediateImages) {
                //Console.WriteLine (ims);
                MPSImageBatch.Synchronize (imBatch, commandBuffer);
            }

            //
            // Setup the completed callback
            //
            commandBuffer.AddCompletedHandler (cmdBuf => {

                //Console.WriteLine ($"{stopwatch.Elapsed} END BATCH {batchIndex} (thread {Thread.CurrentThread.ManagedThreadId})");
                semaphore.Release ();

                if (cmdBuf.Error != null) {
                    Console.WriteLine ("Command Buffer Error: " + cmdBuf.Error.Description);
                }

                //
                // Process the results
                //
                if (returnBatch != null) {
                    var results = returnBatch.ToArray ();
                    foreach (var r in results) {
                        //Console.WriteLine ($"BI{batchIndex} Results handle {r.Handle}");
                        //Console.WriteLine (r.NumberOfImages);
                    }
                }

                //
                // Record history
                //
                //Console.WriteLine ($"{intermediateImages.Length} ims");
                if (intermediateImages.Length - 1 != intermediateHandles.Length) {
                    throw new ApplicationException ("Trained intermediate images without handles");
                }
                var loss = intermediateImages.Length > 0 ?
                    intermediateImages[0].Select (x => new MPSImageTensor (x)).ToArray () :
                    System.Array.Empty<Tensor> ();
                var bh = new Dictionary<string, Tensor[]> ();
                for (var imi = 1; imi < intermediateImages.Length; imi++) {
                    var key = intermediateHandles[imi - 1].Label;
                    bh[key] = intermediateImages[imi].Select (x => new MPSImageTensor (x)).ToArray ();
                }
                recordHistory (new TrainingHistory.BatchHistory (loss, bh));

                //
                // Free the temps
                //
                //var allocBefore = device.GetCurrentAllocatedSize ();

                foreach (var t in temporaryBatchImages) {
                    t.Dispose ();
                }
                temporaryBatchImages = Array.Empty<MPSImage> ();
                foreach (var b in batch) {
                    b.Dispose ();
                }
                batch = Array.Empty<NSArray<MPSImage>> ();

                //var allocAfter = device.GetCurrentAllocatedSize ();
                //Console.WriteLine ($"{stopwatch.Elapsed} ALLOCD {(long)allocBefore-(long)allocAfter:#,0} = {allocBefore:#,0} - {allocAfter:#,0} BATCH {batchIndex} (thread {Thread.CurrentThread.ManagedThreadId})");

            });

            //
            // Run the batch
            //
            commandBuffer.Commit ();

            return commandBuffer;
        }

        (NSArray<MPSImage>[], MPSImage[]) GetBatch (Func<TensorHandle[], IEnumerable<Tensor>> trainingData, int batchSize)
        {
            var temps = new List<MPSImage> ();

            var nsources = sourceHandles.Length;
            var batch = new List<List<MPSImage>> (batchSize);
            for (var i = 0; i < batchSize; i++) {
                //Console.WriteLine ($"GET BATCH IMAGE {i}");
                var data = trainingData (sourceHandles);
                var dataImages = data.Select (ImageFromTensor).ToList ();
                batch.Add (dataImages);
            }

            var sources = new NSArray<MPSImage>[nsources];
            for (var si = 0; si < nsources; si++) {
                var b = new MPSImage[batchSize];
                for (var bi = 0; bi < batchSize; bi++) {
                    b[bi] = batch[bi][si];
                }
                sources[si] = NSArray<MPSImage>.FromNSObjects (b);
            }

            return (sources, temps.ToArray());

            MPSImage ImageFromTensor (Tensor t)
            {
                if (t is MPSImageTensor it) {
                    return t.GetMetalImage (device);
                }
                else {
                    var i = t.GetMetalImage (device);
                    temps.Add (i);
                    return i;
                }
            }
        }
    }
}
