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
    public class InferenceGraph : Graph
    {
        public InferenceGraph (string label, Tensor[] inputs, Tensor[] outputs, MPSNNGraph graph)
            : base (label, graph, inputs, outputs, graph.Device)
        {
        }

        public InferenceGraph (string label, Tensor[] inputs, Tensor[] outputs, EvaluationGraph graph)
            : base (label, graph.MetalGraph, inputs, outputs, graph.Device)
        {
        }

        public InferenceGraph (string label, Tensor[] inputs, Tensor[] outputs, IMTLDevice device)
            : base (label, CreateInferenceGraph (label, outputs, keepDropoutDuringInference: true, device: device), inputs, outputs, device)
        {
        }

        protected static MPSNNGraph CreateInferenceGraph (string label, Tensor[] outputs, bool keepDropoutDuringInference, IMTLDevice device)
        {
            //
            // Build the graph
            //
            var context = new MetalImageNodeContext (label, false, device);

            if (outputs.Length == 0)
                throw new InvalidOperationException ("Cannot create a graph with no outputs");


            //
            // Create the graph
            //
            var outputImageNodes = outputs.Select (x => x.GetMetalImageNode (context)).ToArray ();
            var resultsAreNeeded = outputs.Select (x => true).ToArray ();
            var evalGraph = MPSNNGraph.Create (device, outputImageNodes, resultsAreNeeded);
            evalGraph.Format = MPSImageFeatureChannelFormat.Float32;

            return evalGraph;
        }

        public Tensor[][] Predict (DataSet dataSet, int batchSize, int numBatches)
        {
            using var q = Device.CreateCommandQueue ();
            if (q == null)
                throw new Exception ("Failed to create command queue");

            var semaphore = new Semaphore (2, 2);

            return Predict (dataSet, batchSize, numBatches, semaphore, q);
        }

        public Tensor[][] Predict (DataSet dataSet, int batchSize, int numBatches, Semaphore semaphore, IMTLCommandQueue queue)
        {
            //
            // Refresh weights incase they changed since last time
            //
            MetalGraph.ReloadFromDataSources ();

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
            // Evaluate
            //
            MPSCommandBuffer? lcb = null;
            for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {
                lcb = PredictBatch (dataSet, batchSize, semaphore, queue, AddHistory, batchIndex);
            }
            if (lcb != null) {
                lcb.WaitUntilCompleted ();
            }

            return h.Select (x => x.Results).ToArray ();
        }

        public MPSCommandBuffer PredictBatch (DataSet dataSet, int batchSize, Semaphore semaphore, IMTLCommandQueue queue, Action<TrainingHistory.BatchHistory> recordHistory, int batchIndex)
        {
            return EncodeBatch (batchIndex, dataSet, batchSize, recordHistory, semaphore, queue);
        }

        protected override void OnBatchCompleted (TrainingHistory.BatchHistory batchResults)
        {
            //foreach (var r in batchResults.Results) {
            //    Console.WriteLine (Label + " Result = " + r);
            //}
        }
    }
}
