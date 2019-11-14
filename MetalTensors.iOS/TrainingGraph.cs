using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using Foundation;
using Metal;
using MetalPerformanceShaders;
using MetalTensors.Tensors;

namespace MetalTensors
{
    class TrainingGraph
    {
        readonly IMTLDevice device;
        readonly MPSNNGraph trainingGraph;
        readonly TensorHandle[] sourceHandles;
        readonly LayerHandle[] intermediateHandles;

        //readonly Stopwatch stopwatch = new Stopwatch ();

        public TrainingGraph (Tensor output, IMTLDevice device)
        {
            this.device = device;
            //stopwatch.Start ();

            //
            // Build the training graph
            //
            var thisImageNode = output.GetMetalImageNode (true, device);

            var initialGrad = new MPSNNInitialGradientNode (thisImageNode);
            var lossNodesIndex = new Dictionary<IntPtr, MPSNNForwardLossNode> ();
            var trainingGraphTermini = initialGrad.GetTrainingGraph (null, (gradientNode, inferenceNode, inferenceSource, gradientSource) => {
                //Console.WriteLine ($"gradientNode={gradientNode}, inferenceNode={inferenceNode}, inferenceSource={inferenceSource}, gradientSource={gradientSource}");
                gradientNode.ResultImage.Format = MPSImageFeatureChannelFormat.Float32;
                if (inferenceNode is MPSNNForwardLossNode ln) {
                    lossNodesIndex[ln.Handle] = ln;
                }
            });

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

        public TrainingHistory Train (Func<TensorHandle[], IEnumerable<Tensor>> trainingData, int batchSize, int numBatches)
        {
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
            //stopwatch.Restart ();
            using var q = device.CreateCommandQueue ();

            var semaphore = new Semaphore (2, 2);

            MPSCommandBuffer? lcb = null;
            for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {
                lcb = TrainBatch (batchIndex, trainingData, batchSize, AddHistory, semaphore, q);
            }
            if (lcb != null) {
                lcb.WaitUntilCompleted ();
            }

            return new TrainingHistory (h);
        }

        MPSCommandBuffer TrainBatch (int batchIndex, Func<TensorHandle[], IEnumerable<Tensor>> trainingData, int batchSize, Action<TrainingHistory.BatchHistory> recordHistory, Semaphore semaphore, IMTLCommandQueue queue)
        {
            semaphore.WaitOne ();
            //Console.WriteLine ($"{stopwatch.Elapsed} START BATCH {batchIndex}");

            //
            // Load data
            //
            NSArray<MPSImage>[] batch;
            try {
                batch = GetBatch (trainingData, batchSize);
            }
            catch {
                semaphore.Release ();
                throw;
            }

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

                //Console.WriteLine ($"{stopwatch.Elapsed} END BATCH {batchIndex}");
                semaphore.Release ();

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
                //Console.WriteLine (resultsTensors);
            });

            //
            // Run the batch
            //
            commandBuffer.Commit ();

            return commandBuffer;
        }

        NSArray<MPSImage>[] GetBatch (Func<TensorHandle[], IEnumerable<Tensor>> trainingData, int batchSize)
        {
            var nsources = sourceHandles.Length;
            var batch = new List<List<MPSImage>> (batchSize);
            for (var i = 0; i < batchSize; i++) {
                var data = trainingData (sourceHandles);
                var dataImages = data.Select (x => x.GetMetalImage (device)).ToList ();
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

            return sources;
        }
    }
}
