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

        protected Graph (string label, MPSNNGraph graph, IMTLDevice device)
        {
            this.Device = device;
            Label = label;
            this.MetalGraph = graph;
            graph.Label = Label;
            //stopwatch.Start ();

            sourceHandles = graph.SourceImageHandles.Select (x => (TensorHandle)x).ToArray ();
            //var resultStateHandles = trainingGraph.ResultStateHandles;
            intermediateHandles = graph.IntermediateImageHandles.Select (x => (LayerHandle)x).ToArray ();

            //Console.WriteLine (intermediateHandles);
            //Console.WriteLine (trainingGraph.DebugDescription);
        }

        public override string ToString () => Label;

        protected static MPSNNGraph CreateEvaluationGraph (string label, Tensor trainingOutput, bool keepDropoutDuringInference, out (LayerTensor, LossLayer)[] losses, IMTLDevice device)
        {
            var graphOutputTensor = trainingOutput;
            if (!keepDropoutDuringInference) {
                graphOutputTensor = trainingOutput.RemoveLayers<DropoutLayer> ();
            }

            //
            // Build the training graph
            //
            var context = new MetalImageNodeContext (label, false, device);

            //
            // Find all losses and loss inputs
            //
            var lossesL = new List<(Tensor Output, (LayerTensor, LossLayer) Loss)> ();
            var (flatModel, _) = graphOutputTensor.Model ().Flatten ();
            foreach (var t in flatModel.Tensors) {
                if (t is LayerTensor lt && lt.Layer is LossLayer ll) {
                    var o = lt.Inputs[0];
                    lossesL.Add ((o, (lt, ll)));
                }
            }
            losses = lossesL.Select (x => x.Loss).ToArray ();
            //Console.WriteLine (outputs);
            //Console.WriteLine (losses);

            //
            // Get inputs
            //
            var outputs = new List<Tensor> ();
            foreach (var l in lossesL) {
                var o = l.Output;
                var lossType = l.Loss.Item2.LossType;
                if (lossType == LossType.SigmoidCrossEntropy) {
                    o = o.Sigmoid ();
                }
                else if (lossType == LossType.SoftMaxCrossEntropy) {
                    o = o.SoftMax ();
                }
                outputs.Add (o);
            }

            if (outputs.Count == 0)
                throw new InvalidOperationException ("Cannot create a graph without one or more outputs");


            //
            // Create the graph
            //
            var outputImageNodes = outputs.Select (x => x.GetMetalImageNode (context)).ToArray ();
            var resultsAreNeeded = outputs.Select (x => true).ToArray ();
            var evalGraph = MPSNNGraph.Create (device, outputImageNodes, resultsAreNeeded);
            evalGraph.Format = MPSImageFeatureChannelFormat.Float32;

            return evalGraph;
        }

        public MPSCommandBuffer BeginBatch (int batchIndex, DataSet dataSet, int batchSize, Action<TrainingHistory.BatchHistory> recordHistory, Stopwatch stopwatch, Semaphore semaphore, IMTLCommandQueue queue)
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
                (batch, temporaryBatchImages) = GetBatch (batchIndex, dataSet, batchSize);
            }
            catch {
                semaphore.Release ();
                throw;
            }
            //Console.WriteLine ($"BATCH BYTE SIZE {batchSize*(2+1)*4:#,0}");

            // No using because it is returned
            var commandBuffer = MPSCommandBuffer.Create (queue);

            //
            // Encode the graph
            //
            var intermediateImagesMA = new NSMutableArray<NSArray<MPSImage>> ();
            var destinationStates = new NSMutableArray<NSArray<MPSState>> ();
            NSArray<MPSImage>? returnBatch = MetalGraph.EncodeBatch (commandBuffer, batch, Array.Empty<NSArray<MPSState>> (), intermediateImagesMA, null);
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
                    Console.WriteLine ($"{Label}: Command Buffer Error on batch {batchIndex}: {cmdBuf.Error.Description}");
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
                var h = new TrainingHistory.BatchHistory (results, loss, bh);
                OnBatchCompleted (h);
                recordHistory (h);

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

        protected virtual void OnBatchCompleted (TrainingHistory.BatchHistory batchResults)
        {
        }

        protected virtual TensorHandle[] GetBatchHandles ()
        {
            return sourceHandles;

        }

        (NSArray<MPSImage>[], MPSImage[]) GetBatch (int batchIndex, DataSet dataSet, int batchSize)
        {
            var temps = new List<MPSImage> ();

            var batchHandles = GetBatchHandles ();
            var nsources = batchHandles.Length;
            var batch = new List<MPSImage[]> (batchSize);

            //
            // Map to dataset
            //
            var cols = dataSet.Columns;
            var dataCols = new List<(int HandleIndex, int ColumnIndex)> ();
            var missingHandles = new List<int> (0);
            for (var i = 0; i < batchHandles.Length; i++) {
                var bh = batchHandles[i];
                var col = bh.Label;
                var ci = Array.IndexOf<string> (cols, col);
                if (ci >= 0) {
                    dataCols.Add ((i, ci));
                }
                else {
                    if (bh.Tensor is PlaceholderTensor)
                        missingHandles.Add (i);
                }
            }
            if (missingHandles.Count == 1 && dataCols.Count == 1 && cols.Length == 2) {
                // Auto-match missing column
                var aci = 1 - dataCols[0].ColumnIndex;
                dataCols.Add ((missingHandles[0], aci));
                missingHandles.RemoveAt (0);
            }
            if (missingHandles.Count > 0) {
                var mcols = string.Join (", ", missingHandles.Select (x => "\"" + batchHandles[x].Label + "\""));
                throw new InvalidOperationException ($"Data set is missing required columns {mcols}");
            }

            var constantImages = batchHandles.Select (x => ImageFromTensor (x.Tensor)).ToList ();

            //
            // Get Data
            //
            for (var i = 0; i < batchSize; i++) {
                //Console.WriteLine ($"GET BATCH IMAGE {i}");
                Tensor?[] row = dataSet.GetRow (batchIndex * batchSize + i);
                // Make sure this is a new array
                var rowImages = constantImages.ToArray ();
                foreach (var (hi, ci) in dataCols) {
                    if (ci >= row.Length) {
                        throw new InvalidOperationException ($"Data source returned a row with {row.Length} columns, but {cols.Length} were expected");
                    }
                    var t = row[ci];
                    if (t == null)
                        throw new InvalidOperationException ($"Data source returned a row with a null tensor in column \"{cols[ci]}\"");
                    rowImages[hi] = ImageFromTensor (t);
                }
                batch.Add (rowImages);
            }

            //
            // Sort into batches
            //
            var sources = new NSArray<MPSImage>[nsources];
            for (var si = 0; si < nsources; si++) {
                var b = new MPSImage[batchSize];
                for (var bi = 0; bi < batchSize; bi++) {
                    b[bi] = batch[bi][si];
                }
                sources[si] = NSArray<MPSImage>.FromNSObjects (b);
            }

            return (sources, temps.ToArray ());

            MPSImage ImageFromTensor (Tensor t)
            {
                if (t is MPSImageTensor it) {
                    return t.GetMetalImage (Device);
                }
                else {
                    var i = t.GetMetalImage (Device);
                    temps.Add (i!);
                    return i!;
                }
            }
        }

        protected static void ExportTensor (Tensor tensor, MetalImageNodeContext context)
        {
            var resultImage = tensor.GetMetalImageNode (context);
            resultImage.ExportFromGraph = true;
            resultImage.SynchronizeResource = true;
            resultImage.ImageAllocator = MPSImage.DefaultAllocator;
        }
    }
}
