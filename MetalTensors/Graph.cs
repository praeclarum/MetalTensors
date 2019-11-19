﻿using System;
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
    abstract class Graph
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
            // Export all losses and loss inputs
            //
            var outputs = new List<Tensor> ();
            var lossesL = new List<(LayerTensor, LossLayer)> ();
            var (flatModel, _) = graphOutputTensor.Model ().Flatten ();
            foreach (var t in flatModel.Tensors) {
                if (t is LayerTensor lt && lt.Layer is LossLayer ll) {
                    var o = lt.Inputs[0];
                    outputs.Add (o);
                    lossesL.Add ((lt, ll));
                }
            }
            losses = lossesL.ToArray ();
            Console.WriteLine (outputs);
            Console.WriteLine (losses);

            //
            // Create the graph
            //
            var outputImageNodes = outputs.Select (x => x.GetMetalImageNode (context)).ToArray ();
            var resultsAreNeeded = outputs.Select (x => true).ToArray ();
            var evalGraph = MPSNNGraph.Create (device, outputImageNodes, resultsAreNeeded);
            evalGraph.Format = MPSImageFeatureChannelFormat.Float32;

            return evalGraph;
        }

        public MPSCommandBuffer BeginBatch (int batchIndex, LoadBatch trainingData, int batchSize, Action<TrainingHistory.BatchHistory> recordHistory, Stopwatch stopwatch, Semaphore semaphore, IMTLCommandQueue queue)
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

            // No using because it is returned
            var commandBuffer = MPSCommandBuffer.Create (queue);

            //
            // Encode the graph
            //
            var intermediateImagesMA = new NSMutableArray<NSArray<MPSImage>> ();
            var destinationStates = new NSMutableArray<NSArray<MPSState>> ();
            NSArray<MPSImage>? returnBatch = MetalGraph.EncodeBatch (commandBuffer, batch, System.Array.Empty<NSArray<MPSState>> (), intermediateImagesMA, null);
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

        (NSArray<MPSImage>[], MPSImage[]) GetBatch (LoadBatch trainingData, int batchSize)
        {
            var temps = new List<MPSImage> ();

            var batchHandles = GetBatchHandles ();
            var nsources = batchHandles.Length;
            var batch = new List<List<MPSImage>> (batchSize);
            for (var i = 0; i < batchSize; i++) {
                //Console.WriteLine ($"GET BATCH IMAGE {i}");
                var data = trainingData (batchHandles);
                var dataImages = data.Select (ImageFromTensor).ToList ();
                if (dataImages.Count != batchHandles.Length)
                    throw new InvalidOperationException ($"{batchHandles.Length} tensors are needed to train, {dataImages.Count} provided");
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

            return (sources, temps.ToArray ());

            MPSImage ImageFromTensor (Tensor t)
            {
                if (t is MPSImageTensor it) {
                    return t.GetMetalImage (Device);
                }
                else {
                    var i = t.GetMetalImage (Device);
                    temps.Add (i);
                    return i;
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
