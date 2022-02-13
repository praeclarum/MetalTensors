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
    public class EvaluationGraph : Graph
    {
        public (LayerTensor Tensor, LossLayer Layer)[] Losses { get; }

        public EvaluationGraph (string label, Tensor trainingOutput, bool keepDropoutDuringInference, IMTLDevice device)
            : base (label, CreateEvaluationGraph (label, trainingOutput, keepDropoutDuringInference, out var losses, device), device)
        {
            Losses = losses;
        }

        public TrainingHistory Evaluate (DataSet dataSet, int batchSize, int numBatches)
        {
            using var q = Device.CreateCommandQueue ();
            if (q == null)
                throw new Exception ("Failed to create command queue");

            var semaphore = new Semaphore (2, 2);

            return Evaluate (dataSet, batchSize, numBatches, semaphore, q);
        }

        public TrainingHistory Evaluate (DataSet dataSet, int batchSize, int numBatches, Semaphore semaphore, IMTLCommandQueue queue)
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
            // Train
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

        protected override TensorHandle[] GetBatchHandles ()
        {
            //
            // Add Labels
            //
            var r = base.GetBatchHandles ().ToList ();

            r.AddRange (Losses.Select (x => x.Tensor.Inputs[1].Handle));

            return r.ToArray ();
        }

        protected override void OnBatchCompleted (TrainingHistory.BatchHistory batchResults)
        {
            //foreach (var r in batchResults.Results) {
            //    Console.WriteLine (Label + " Result = " + r);
            //}
        }
    }
}
