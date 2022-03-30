using System;
using System.Collections.Concurrent;
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
        protected readonly IMTLCommandQueue modelQueue;
        protected readonly Semaphore modelSemaphore;

        readonly TensorHandle[] sourceHandles;
        protected readonly LayerHandle[] intermediateHandles;
        readonly int[] sourceToInputMap;
        readonly int[] sourceToOutputMap;
        MPSImage?[]? madeStaticImages;
        readonly Tensor[] modelInputs;
        readonly Tensor[] modelOutputs;

        protected int nextCommandId = 0;

        protected Graph (string label, MPSNNGraph graph, Tensor[] inputs, Tensor[] outputs, IMTLCommandQueue queue, Semaphore semaphore)
        {
            this.Device = queue.Device;
            Label = label;
            this.MetalGraph = graph;
            graph.Label = Label;
            this.modelInputs = inputs;
            this.modelOutputs = outputs;

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

            //
            // Queue and Semaphore
            //
            modelQueue = queue;
            modelSemaphore = semaphore;

            //Console.WriteLine (graph.DebugDescription);
        }

        public override string ToString () => Label;

        public MPSCommandBuffer EncodeBatch (int batchIndex, DataSet dataSet, int batchSize, Action<TrainingHistory.BatchHistory> recordHistory)
        {
            var (inputs, outputs) = dataSet.GetBatch (batchIndex * batchSize, batchSize, Device);
            return EncodeBatch (inputs, outputs, batchSize, recordHistory);
        }

        public MPSCommandBuffer EncodeBatch (Tensor[][] inputs, Tensor[][] outputs, int batchSize, Action<TrainingHistory.BatchHistory> recordHistory)
        {
            if (inputs.Length < 1)
                throw new ArgumentException ($"At least one input is needed in a batch");

            //
            // This pool is necessary for Metal to clean up its objects
            //
            using var pool = new NSAutoreleasePool ();

            // No using because it is returned
            var commandBuffer = MPSCommandBuffer.Create (modelQueue);
            commandBuffer.Label = $"{Label} {nextCommandId}";
            Interlocked.Increment (ref nextCommandId);

            //
            // Load data
            //
            //var (batch, temporaryBatchImages) = GetSourceImages (inputs, outputs);
            var batchSourceImages = RentSourceImages (batchSize);
            var batch = EncodeSourceImages (inputs, outputs, batchSourceImages, batchSize, commandBuffer);

            //Console.WriteLine ($"BATCH BYTE SIZE {batchSize*(2+1)*4:#,0}");
            //Console.WriteLine ($"{stopwatch.Elapsed} START BATCH {batchIndex} (thread {Thread.CurrentThread.ManagedThreadId})");

            //
            // Wait for the last command to finish
            //
            modelSemaphore.WaitOne ();

            //
            // Encode the graph
            //
            var intermediateImagesMA = new NSMutableArray<NSArray<MPSImage>> ();
            var destinationStates = new NSMutableArray<NSArray<MPSState>> ();
            var returnBatch = MetalGraph.EncodeBatch (commandBuffer, batch, null, intermediateImagesMA, null);
            if (intermediateImagesMA.Count > 0)
                throw new Exception ("Not expecting intermediate images");

            //
            // Synchronize needed images
            //
            if (returnBatch != null) {
                MPSImageBatch.Synchronize (returnBatch, commandBuffer);
            }

            //
            // Setup the completed callback
            //
            commandBuffer.AddCompletedHandler (cmdBuf => {

                ReturnSourceImages (batchSourceImages);

                //Console.WriteLine ($"{stopwatch.Elapsed} END BATCH {batchIndex} (thread {Thread.CurrentThread.ManagedThreadId})");
                modelSemaphore.Release ();

                if (cmdBuf.Error != null) {
                    Console.WriteLine ($"{Label}: Command Buffer Error: {cmdBuf.Error.Description}");
                }

                //
                // Record results
                //
                var results = Array.Empty<Tensor> ();                
                if (returnBatch != null) {
                    var returnBatchCount = returnBatch.Count;
                    results = new Tensor[returnBatchCount];
                    for (nuint bi = 0; bi < returnBatchCount; bi++) {
                        var image = returnBatch.GetItem<MPSImage> (bi);
                        var ih = image.Height;
                        var iw = image.Width;
                        var ic = image.FeatureChannels;
                        var n = ih * iw * ic;
                        if (n == 1) {
                            unsafe {
                                var vs = stackalloc float[1];
                                image.ReadBytes ((IntPtr)vs, MPSDataLayout.HeightPerWidthPerFeatureChannels, 0);
                                results[bi] = new ConstantTensor (vs[0], Shapes.GetShape (ih, iw, ic));
                            }
                            image.Dispose ();
                        }
                        else {
                            results[bi] = new MPSImageTensor (image);
                        }                            
                    }
                    returnBatch.Dispose ();
                }

                //
                // Record the intermediates
                //
                var bh = new Dictionary<string, Tensor[]> ();
                //
                // Broadcast the results to whomever is listening
                //
                var h = new TrainingHistory.BatchHistory (results, new Dictionary<string, float>(), bh, cmdBuf.Device);
                recordHistory (h);
            });

            //
            // Run the batch
            //
            commandBuffer.Commit ();

            return commandBuffer;
        }

        readonly ConcurrentDictionary<int, ConcurrentBag<MPSImage[][]>> availableSourceImages = new ConcurrentDictionary<int, ConcurrentBag<MPSImage[][]>> ();

        protected MPSImage[][] RentSourceImages (int batchSize)
        {
            var numSources = sourceHandles.Length;
            MPSImage[][] images;
            if (availableSourceImages.TryGetValue (batchSize, out var imageCache)) {
                if (imageCache.TryTake (out images)) {
                    return images;
                }
                else {
                    images = new MPSImage[numSources][];
                }
            }
            else {
                availableSourceImages.TryAdd (batchSize, new ConcurrentBag<MPSImage[][]> ());
                images = new MPSImage[numSources][];
            }

            var statics = GetStaticImages ();
            for (var si = 0; si < numSources; si++) {
                images[si] = new MPSImage[batchSize];
                var c = statics[si];
                if (c != null) {
                    for (var bi = 0; bi < batchSize; bi++) {
                        images[si][bi] = c;
                    }
                }
                else {
                    for (var bi = 0; bi < batchSize; bi++) {
                        images[si][bi] = sourceHandles[si].Tensor.CreateUninitializedImage ();
                    }
                }
            }
            return images;
        }

        protected void ReturnSourceImages (MPSImage[][] images)
        {
            try {
                var batchSize = images[0].Length;
                if (availableSourceImages.TryGetValue (batchSize, out var imageCache)) {
                    imageCache.Add (images);
                }
            }
            catch (Exception ex) {
                Console.WriteLine ($"Failed to return image: {ex}");
            }
        }

        protected NSArray<MPSImage>[] CopySourceImages (Tensor[][] inputs, Tensor[][] outputs, MPSImage[][] images, IMTLCommandQueue queue)
        {
            var batchSize = inputs.Length;

            var statics = GetStaticImages ();
            var ns = sourceHandles.Length;

            var tasks = new List<Task> ();

            for (var bi = 0; bi < batchSize; bi++) {
                for (var si = 0; si < ns; si++) {
                    if (statics[si] != null)
                        continue;
                    var inputIndex = sourceToInputMap[si];
                    if (0 <= inputIndex && inputIndex < inputs[bi].Length) {
                        var image = images[si][bi];
                        tasks.Add (inputs[bi][inputIndex].CopyToAsync (image, queue));
                    }
                    else {
                        var outputIndex = sourceToOutputMap[si];
                        if (0 <= outputIndex && outputIndex < outputs[bi].Length) {
                            var image = images[si][bi];
                            tasks.Add (outputs[bi][outputIndex].CopyToAsync (image, queue));
                        }
                        else {
                            throw new KeyNotFoundException ($"Cannot find data for {sourceHandles[si].Label}. The model expects {modelInputs.Length} inputs and {modelOutputs.Length} outputs. Provided data has {inputs[0].Length} inputs and {outputs[0].Length} outputs.");
                        }
                    }
                }
            }

            Task.WaitAll (tasks.ToArray ());

            var imageArrays = new NSArray<MPSImage>[ns];
            for (var si = 0; si < ns; si++) {
                imageArrays[si] = NSArray<MPSImage>.FromNSObjects (images[si]);
            }
            return imageArrays;
        }

        protected NSArray<MPSImage>[] EncodeSourceImages (Tensor[][] inputs, Tensor[][] outputs, MPSImage[][] images, int batchSize, MPSCommandBuffer commands)
        {
            var statics = GetStaticImages ();
            var ns = sourceHandles.Length;

            for (var bi = 0; bi < batchSize; bi++) {
                for (var si = 0; si < ns; si++) {
                    if (statics[si] != null)
                        continue;
                    var inputIndex = sourceToInputMap[si];
                    if (0 <= inputIndex && inputIndex < inputs[bi].Length) {
                        var image = images[si][bi];
                        inputs[bi][inputIndex].EncodeToCommandBuffer (image, commands);
                    }
                    else {
                        var outputIndex = sourceToOutputMap[si];
                        if (0 <= outputIndex && outputIndex < outputs[bi].Length) {
                            var image = images[si][bi];
                            outputs[bi][outputIndex].EncodeToCommandBuffer (image, commands);
                        }
                        else {
                            throw new KeyNotFoundException ($"Cannot find data for {sourceHandles[si].Label}. The model expects {modelInputs.Length} inputs and {modelOutputs.Length} outputs. Provided data has {inputs[0].Length} inputs and {outputs[0].Length} outputs.");
                        }
                    }
                }
            }

            var imageArrays = new NSArray<MPSImage>[ns];
            for (var si = 0; si < ns; si++) {
                imageArrays[si] = NSArray<MPSImage>.FromNSObjects (images[si]);
            }
            return imageArrays;
        }

        protected (NSArray<MPSImage>[] SourceImages, MPSImage[] TemporaryImages) GetSourceImages (Tensor[][] inputs, Tensor[][] outputs)
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
                                throw new KeyNotFoundException ($"Cannot find data for {sourceHandles[si].Label}. The model expects {modelInputs.Length} inputs and {modelOutputs.Length} outputs. Provided data has {inputs[0].Length} inputs and {outputs[0].Length} outputs.");
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
