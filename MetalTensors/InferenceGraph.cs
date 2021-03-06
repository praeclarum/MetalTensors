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
    class InferenceGraph : Graph
    {
        public InferenceGraph (string label, MPSNNGraph graph)
            : base (label, graph, graph.Device)
        {
        }

        public TrainingHistory Predict (DataSet dataSet, int batchSize, int numBatches)
        {
            using var q = Device.CreateCommandQueue ();
            if (q == null)
                throw new Exception ("Failed to create command queue");

            var semaphore = new Semaphore (2, 2);

            return Predict (dataSet, batchSize, numBatches, semaphore, q);
        }

        public TrainingHistory Predict (DataSet dataSet, int batchSize, int numBatches, Semaphore semaphore, IMTLCommandQueue queue)
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
            var stopwatch = new Stopwatch ();
            stopwatch.Restart ();

            MPSCommandBuffer? lcb = null;
            for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {
                lcb = BeginBatch (batchIndex, dataSet, batchSize, AddHistory, stopwatch, semaphore, queue);
            }
            if (lcb != null) {
                lcb.WaitUntilCompleted ();
            }

            return new TrainingHistory (h);
        }

        protected override void OnBatchCompleted (TrainingHistory.BatchHistory batchResults)
        {
            //foreach (var r in batchResults.Results) {
            //    Console.WriteLine (Label + " Result = " + r);
            //}
        }
    }
}
