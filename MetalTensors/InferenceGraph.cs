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
    public class InferenceGraph : Graph
    {
        bool needsReloadWeights = true;

        public InferenceGraph (string label, Tensor[] inputs, Tensor[] outputs, IMTLCommandQueue queue, Semaphore semaphore)
            : base (label, CreateInferenceGraph (label, outputs, device: queue.Device), inputs, outputs, queue, semaphore)
        {
        }

        protected static MPSNNGraph CreateInferenceGraph (string label, Tensor[] outputs, IMTLDevice device)
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
            var outputImageNodes = outputs.Select (x => x.GetImageNode (context)).ToArray ();
            var resultsAreNeeded = outputs.Select (x => true).ToArray ();
            var infGraph = MPSNNGraph.Create (device, outputImageNodes, resultsAreNeeded);
            infGraph.Format = MPSImageFeatureChannelFormat.Float32;

            return infGraph;
        }

        public Tensor[][] Predict (DataSet dataSet, int batchSize, int numBatches)
        {
            using var pool = new NSAutoreleasePool ();

            if (needsReloadWeights) {
                MetalGraph.ReloadFromDataSources ();
                needsReloadWeights = false;
            }

            //
            // Init history
            //
            var h = new List<Tensor[]> ();
            void AddHistory (TrainingHistory.BatchHistory bh)
            {
                lock (h) {
                    h.Add (bh.Results);
                }
            }

            //
            // Evaluate
            //
            MPSCommandBuffer? lcb = null;
            for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {
                lcb = EncodeBatch (batchIndex, dataSet, batchSize, AddHistory);
            }
            if (lcb != null) {
                lcb.WaitUntilCompleted ();
            }

            return h.ToArray ();
        }

        public Tensor[][] Predict (Tensor[][] inputsBatch, int batchSize)
        {
            using var pool = new NSAutoreleasePool ();

            if (needsReloadWeights) {
                MetalGraph.ReloadFromDataSources ();
                needsReloadWeights = false;
            }

            //
            // Init history
            //
            var h = new Tensor[batchSize][];
            void AddHistory (TrainingHistory.BatchHistory bh)
            {
                var r = bh.Results;
                for (var bi = 0; bi < r.Length; bi++) {
                    h[bi] = new[] { r[bi] };
                }
            }

            //
            // Evaluate
            //
            MPSCommandBuffer lcb = EncodeBatch (inputsBatch, Array.Empty<Tensor[]>(), batchSize, AddHistory);
            if (lcb != null) {
                lcb.WaitUntilCompleted ();
            }

            return h;
        }

        public void SetNeedsReloadWeights ()
        {
            needsReloadWeights = true;
        }
    }
}
