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
    public class TrainingGraph : Graph
    {
        readonly (ConvDataSource Weights, bool Trainable)[] convWeights;
        readonly EvaluationGraph evalGraph;

        public TrainingGraph (string label, Tensor[] inputs, Tensor[] outputs, Tensor[] losses, Dictionary<Layer, bool> trainable, EvaluationGraph evalGraph, IMTLDevice device)
            : base (label, CreateTrainingGraph (label, losses, trainable, device, out var cweights), inputs, outputs, device)
        {
            convWeights = cweights;
            this.evalGraph = evalGraph;
        }

        static MPSNNGraph CreateTrainingGraph (string label, Tensor[] losses, Dictionary<Layer, bool> trainable, IMTLDevice device, out (ConvDataSource Weights, bool Trainable)[] cweights)
        {
            if (losses.Length < 1) {
                throw new ArgumentException ("Loss is required in order to train", nameof(losses));
            }
            //stopwatch.Start ();

            //
            // Build the training graph
            //
            var context = new MetalImageNodeContext (label, false, device);
            var trainingOutput = losses.Length == 1 ? losses[0] : Tensor.Sum (losses);
            var thisImageNode = trainingOutput.GetMetalImageNode (context);

            var initialGrad = new MPSNNInitialGradientNode (thisImageNode);
            var convWeightsL = new List<(ConvDataSource, bool)> ();

            var trainingGraphTermini = initialGrad.GetTrainingGraph (null, (gradientNode, inferenceNode, inferenceSource, gradientSource) => {
                //Console.WriteLine ($"gradientNode={gradientNode}, inferenceNode={inferenceNode}, inferenceSource={inferenceSource}, gradientSource={gradientSource}");
                gradientNode.ResultImage.Format = MPSImageFeatureChannelFormat.Float32;
                if (inferenceNode.ResultImage.MPSHandle is LayerHandle lh &&
                         lh.Layer.GetMetalConvDataSource (device) is ConvDataSource cw) {
                    if (!trainable.ContainsKey (lh.Layer)) {
                        throw new Exception ($"Cannot tell if {lh.Layer} is trainable");
                    }
                    var train = trainable[lh.Layer];
                    convWeightsL.Add ((cw, train));
                    
                    //Console.WriteLine (lh);
                }                
            });

            cweights = convWeightsL.ToArray ();

            var trainingGraphTerminiImageNodes = trainingGraphTermini.Select (x => x.ResultImage).ToArray ();
            var resultsNeeded = trainingGraphTerminiImageNodes.Select (x => true).ToArray ();

            //
            // Export all losses and loss inputs
            //
            foreach (var t in losses) {
                ExportTensor (t, context);
            }

            //
            // Create the graph
            //
            var trainingGraph = MPSNNGraph.Create (device, trainingGraphTerminiImageNodes, resultsNeeded);
            trainingGraph.Format = MPSImageFeatureChannelFormat.Float32;

            return trainingGraph;
        }

        public TrainingHistory Fit (DataSet dataSet, Optimizer optimizer, int batchSize, int numBatches, int validateInterval)
        {
            if (validateInterval <= 0)
                throw new ArgumentException ($"Invalidate validation interval ({validateInterval}) specified.");

            //
            // Set the learning rate
            //
            foreach (var c in convWeights) {
                c.Weights.SetOptimizationOptions (c.Trainable, optimizer.LearningRate);
            }

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
            using var q = Device.CreateCommandQueue ();
            if (q == null)
                throw new Exception ("Failed to create command queue");

            var semaphore = new Semaphore (2, 2);

            MPSCommandBuffer? lcb = null;
            for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {
                lcb = BeginBatch (batchIndex, dataSet, batchSize, AddHistory, stopwatch, semaphore, q);

                if ((batchIndex + 1) % validateInterval == 0) {
                    lcb?.WaitUntilCompleted ();
                    lcb = null;
                    var evalHistory = evalGraph.Evaluate (dataSet, batchSize, 1, semaphore, q);
                    //Console.WriteLine (evalHistory);
                }
            }
            lcb?.WaitUntilCompleted ();

            return new TrainingHistory (h);
        }
    }
}
