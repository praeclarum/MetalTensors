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
    class InferenceGraph : Graph
    {
        public InferenceGraph (string label, MPSNNGraph graph)
            : base (label, graph, graph.Device)
        {
        }

        public TrainingHistory Predict (LoadBatch trainingData, int batchSize, int numBatches)
        {
            using var q = Device.CreateCommandQueue ();

            var semaphore = new Semaphore (2, 2);

            return Predict (trainingData, batchSize, numBatches, semaphore, q);
        }

        public TrainingHistory Predict (LoadBatch trainingData, int batchSize, int numBatches, Semaphore semaphore, IMTLCommandQueue queue)
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
            // Evaluate
            //
            var stopwatch = new Stopwatch ();
            stopwatch.Restart ();

            MPSCommandBuffer? lcb = null;
            for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {
                lcb = BeginBatch (batchIndex, trainingData, batchSize, AddHistory, stopwatch, semaphore, queue);
            }
            if (lcb != null) {
                lcb.WaitUntilCompleted ();
            }

            return new TrainingHistory (h);
        }

        protected override void OnBatchCompleted (TrainingHistory.BatchHistory batchResults)
        {
            foreach (var r in batchResults.Results) {
                Console.WriteLine (Label + " " + r);
            }
        }
    }
}
