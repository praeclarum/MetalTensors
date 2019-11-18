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
    class EvaluationGraph : Graph
    {
        public EvaluationGraph (Tensor trainingOutput, bool ignoreDropoutDuringInference, IMTLDevice device)
            : base (CreateEvaluationGraph (trainingOutput, ignoreDropoutDuringInference, device), device)
        {
        }

        static MPSNNGraph CreateEvaluationGraph (Tensor output, bool ignoreDropoutDuringInference, IMTLDevice device)
        {
            //
            // Build the training graph
            //
            var thisImageNode = output.GetMetalImageNode (true, device);

            var evalGraph = MPSNNGraph.Create (device, thisImageNode, resultIsNeeded: true);
            evalGraph.Format = MPSImageFeatureChannelFormat.Float32;

            return evalGraph;
        }

        public TrainingHistory Evaluate (LoadBatch trainingData, int batchSize, int numBatches)
        {
            using var q = Device.CreateCommandQueue ();

            var semaphore = new Semaphore (2, 2);

            return Evaluate (trainingData, batchSize, numBatches, semaphore, q);
        }

        public TrainingHistory Evaluate (LoadBatch trainingData, int batchSize, int numBatches, Semaphore semaphore, IMTLCommandQueue queue)
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
    }
}
