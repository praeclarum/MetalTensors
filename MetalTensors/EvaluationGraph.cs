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
        public Tensor[] Losses { get; }

        public EvaluationGraph (string label, Tensor[] inputs, Tensor[] outputs, Tensor[] losses, bool keepDropoutDuringInference, IMTLCommandQueue queue, Semaphore semaphore)
            : base (label, CreateEvaluationGraph (label, losses, keepDropoutDuringInference, queue.Device), inputs, outputs, queue, semaphore)
        {
            Losses = losses;
        }

        protected static MPSNNGraph CreateEvaluationGraph(string label, Tensor[] losses, bool keepDropoutDuringInference, IMTLDevice device)
        {
            if (!keepDropoutDuringInference) {
                losses = losses.Select(x => x.RemoveLayers<DropoutLayer>()).ToArray();
            }

            //
            // Build the evaluation graph
            //
            var context = new MetalImageNodeContext(label, false, device);

            var outputs = losses;

            if (outputs.Length == 0)
                throw new InvalidOperationException("Cannot create an evaluation graph without losses");


            //
            // Create the graph
            //
            var outputImageNodes = outputs.Select(x => x.GetImageNode(context)).ToArray();
            var resultsAreNeeded = outputs.Select(x => true).ToArray();
            if (MPSNNGraph.Create(device, outputImageNodes, resultsAreNeeded) is { } evalGraph)
            {
                evalGraph.Format = MPSImageFeatureChannelFormat.Float32;

                return evalGraph;
            }
            else
            {
                throw new Exception("Failed to create evaluation graph");
            }
        }

        public TrainingHistory Evaluate (DataSet dataSet, int batchSize, int numBatches)
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
            // Run
            //
            MPSCommandBuffer? lcb = null;
            for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {
                lcb = EncodeBatch (batchIndex, dataSet, batchSize, AddHistory);
            }
            if (lcb != null) {
                lcb.WaitUntilCompleted ();
            }

            return new TrainingHistory (h);
        }
    }
}
