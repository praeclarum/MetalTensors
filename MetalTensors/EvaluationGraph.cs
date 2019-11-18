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
        public EvaluationGraph (string label, Tensor trainingOutput, bool ignoreDropoutDuringInference, IMTLDevice device)
            : base (label, CreateEvaluationGraph (label, trainingOutput, ignoreDropoutDuringInference, device), device)
        {
        }

        static MPSNNGraph CreateEvaluationGraph (string label, Tensor trainingOutput, bool ignoreDropoutDuringInference, IMTLDevice device)
        {
            //
            // Build the training graph
            //
            var context = new MetalImageNodeContext (label, false, device);
            var thisImageNode = trainingOutput.GetMetalImageNode (context);

            //
            // Export all losses and loss inputs
            //
            var (flatModel, _) = trainingOutput.Model ().Flatten ();
            foreach (var t in flatModel.Tensors) {
                if (t is LayerTensor lt && lt.Layer is LossLayer ll) {
                    ExportTensor (t, context);
                }
            }

            //
            // Create the graph
            //
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

        protected override void OnBatchCompleted (TrainingHistory.BatchHistory batchResults)
        {
            //foreach (var r in batchResults.Results) {
            //    Console.WriteLine (Label + " " + r);
            //}
        }
    }
}
