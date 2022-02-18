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
    public abstract class Graph
    {
        public IMTLDevice Device { get; }

        public string Label { get; }
        public MPSNNGraph MetalGraph { get; }
        readonly TensorHandle[] sourceHandles;
        readonly LayerHandle[] intermediateHandles;
        readonly int[] sourceToInputMap;
        readonly int[] sourceToOutputMap;
        MPSImage?[]? madeStaticImages;

        protected Graph (string label, MPSNNGraph graph, Tensor[] inputs, Tensor[] outputs, IMTLDevice device)
        {
            this.Device = device;
            Label = label;
            this.MetalGraph = graph;
            graph.Label = Label;

            sourceHandles = graph.SourceImageHandles.Select (x => (TensorHandle)x).ToArray ();
            var ns = sourceHandles.Length;
            //var resultStateHandles = trainingGraph.ResultStateHandles;
            intermediateHandles = graph.IntermediateImageHandles.Select (x => (LayerHandle)x).ToArray ();
            //Console.WriteLine (intermediateHandles);

            sourceToInputMap = new int[ns];
            for (var si = 0; si < ns; si++) {
                var h = sourceHandles[si];
                sourceToInputMap[si] = Array.FindIndex (inputs, x => x.Id == h.Id);
            }

            sourceToOutputMap = new int[ns];
            for (var si = 0; si < ns; si++) {
                var h = sourceHandles[si];
                if (h is LabelsHandle lh) {
                    var output = lh.OutputTensor;
                    sourceToOutputMap[si] = Array.FindIndex (outputs, x => x.Id == output.Id);
                }
                else {
                    sourceToOutputMap[si] = -1;
                }
            }

            //Console.WriteLine (graph.DebugDescription);
        }

        public override string ToString () => Label;

        public MPSCommandBuffer EncodeBatch (int batchIndex, DataSet dataSet, int batchSize, Action<TrainingHistory.BatchHistory> recordHistory, Semaphore semaphore, IMTLCommandQueue queue)
        {
            var (inputs, outputs) = dataSet.GetBatch (batchIndex * batchSize, batchSize, queue.Device);
            return EncodeBatch (inputs, outputs, recordHistory, semaphore, queue);
        }

        public MPSCommandBuffer EncodeBatch (Tensor[][] inputs, Tensor[][] outputs, Action<TrainingHistory.BatchHistory> recordHistory, Semaphore semaphore, IMTLCommandQueue queue)
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
            var intermediateImagesMA = new NSMutableArray<NSArray<MPSImage>> ();
            var destinationStates = new NSMutableArray<NSArray<MPSState>> ();
            NSArray<MPSImage>? returnBatch = MetalGraph.EncodeBatch (commandBuffer, batch, null, intermediateImagesMA, null);
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
                    Console.WriteLine ($"{Label}: Command Buffer Error: {cmdBuf.Error.Description}");
                }

                //
                // Record results
                //
                var results = returnBatch != null && returnBatch.Count > 0 ?
                    returnBatch.Select (x => (Tensor)new MPSImageTensor (x)).ToArray () :
                    Array.Empty<Tensor> ();

                //
                // Record the loss history
                //
                //Console.WriteLine ($"{intermediateImages.Length} ims");                
                var loss = intermediateImages.Length > 0 ?
                    intermediateImages[0].Select (x => new MPSImageTensor (x)).ToArray () :
                    Array.Empty<Tensor> ();

                //
                // Record the intermediates
                //
                var bh = new Dictionary<string, Tensor[]> ();
                if (intermediateImages.Length > 0) {
                    if (intermediateImages.Length == intermediateHandles.Length) {
                        for (var imi = 0; imi < intermediateImages.Length; imi++) {
                            var key = intermediateHandles[imi].Label;
                            bh[key] = intermediateImages[imi].Select (x => new MPSImageTensor (x)).ToArray ();
                        }
                    }
                    else if (intermediateImages.Length - 1 != intermediateHandles.Length) {
                        Console.WriteLine ($"! Intermediate images without handles {intermediateImages.Length} vs {intermediateHandles.Length}");
                    }
                    else {
                        for (var imi = 1; imi < intermediateImages.Length; imi++) {
                            var key = intermediateHandles[imi - 1].Label;
                            bh[key] = intermediateImages[imi].Select (x => new MPSImageTensor (x)).ToArray ();
                        }
                    }
                }

                //
                // Broadcast the results to whomever is listening
                //
                var h = new TrainingHistory.BatchHistory (results, loss, bh, temporaryBatchImages, cmdBuf.Device);
                OnBatchCompleted (h);
                recordHistory (h);
            });

            //
            // Run the batch
            //
            commandBuffer.Commit ();

            return commandBuffer;
        }

        protected virtual void OnBatchCompleted (TrainingHistory.BatchHistory batchResults)
        {
        }

        (NSArray<MPSImage>[] SourceImages, MPSImage[] TemporaryImages) GetSourceImages (Tensor[][] inputs, Tensor[][] outputs)
        {
            var batchSize = inputs.Length;
            var temps = new List<MPSImage> ();

            var statics = GetStaticImages ();
            var ns = sourceHandles.Length;
            var images = new MPSImage[ns][];
            for (var si = 0; si < ns; si++) {
                images[si] = new MPSImage[batchSize];
                var c = statics[si];
                if (c != null) {
                    for (var bi = 0; bi < batchSize; bi++) {
                        images[si][bi] = c;
                    }
                }
            }

            for (var bi = 0; bi < batchSize; bi++) {
                for (var si = 0; si < ns; si++) {
                    if (images[si][bi] is null) {
                        var inputIndex = sourceToInputMap[si];
                        if (0 <= inputIndex && inputIndex < inputs[bi].Length) {
                            var image = inputs[bi][inputIndex].GetMetalImage (Device);
                            if (image == null || image.Handle == IntPtr.Zero)
                                throw new Exception ($"Failed to get metal image for {inputs[bi][inputIndex]}");
                            temps.Add (image);
                            images[si][bi] = image;
                        }
                        else {
                            var outputIndex = sourceToOutputMap[si];
                            if (0 <= outputIndex && outputIndex < outputs[bi].Length) {
                                var image = outputs[bi][outputIndex].GetMetalImage (Device);
                                if (image == null || image.Handle == IntPtr.Zero)
                                    throw new Exception ($"Failed to get metal image for {outputs[bi][outputIndex]}");
                                temps.Add (image);
                                images[si][bi] = image;
                            }
                            else {
                                throw new KeyNotFoundException ($"Cannot find data for {sourceHandles[si].Label}");
                            }
                        }
                    }
                }
            }

            var imageArrays = new NSArray<MPSImage>[ns];
            for (var si = 0; si < ns; si++) {
                imageArrays[si] = NSArray<MPSImage>.FromNSObjects (images[si]);
            }

            return (imageArrays, temps.ToArray ());
        }

        protected static void ExportTensor (Tensor tensor, MetalImageNodeContext context)
        {
            var resultImage = tensor.GetImageNode (context);
            resultImage.ExportFromGraph = true;
            resultImage.SynchronizeResource = true;
            resultImage.ImageAllocator = MPSImage.DefaultAllocator;
        }

        MPSImage?[] GetStaticImages ()
        {
            if (madeStaticImages != null)
                return madeStaticImages;

            var images = new MPSImage?[sourceHandles.Length];
            for (var si = 0; si < sourceHandles.Length; si++) {
                var t = sourceHandles[si].Tensor;
                if (t.IsStatic) {
                    images[si] = t.GetMetalImage (Device);
                }
            }

            madeStaticImages = images;
            return images;
        }
    }
}
