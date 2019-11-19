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
    class TrainingGraph : Graph
    {
        readonly (ConvDataSource Weights, bool Trainable)[] convWeights;
        readonly EvaluationGraph evalGraph;

        public TrainingGraph (string label, Tensor output, Dictionary<Layer, bool> trainable, EvaluationGraph evalGraph, IMTLDevice device)
            : base (label, CreateTrainingGraph (label, output, trainable, device, out var cweights), device)
        {
            convWeights = cweights;
            this.evalGraph = evalGraph;
        }

        static MPSNNGraph CreateTrainingGraph (string label, Tensor trainingOutput, Dictionary<Layer, bool> trainable, IMTLDevice device, out (ConvDataSource Weights, bool Trainable)[] cweights)
        {
            //stopwatch.Start ();

            //
            // Build the training graph
            //
            var context = new MetalImageNodeContext (label, false, device);
            var thisImageNode = trainingOutput.GetMetalImageNode (context);

            var initialGrad = new MPSNNInitialGradientNode (thisImageNode);
            var lossNodesIndex = new Dictionary<IntPtr, MPSNNForwardLossNode> ();
            var convWeightsL = new List<(ConvDataSource, bool)> ();
            var trainingGraphTermini = initialGrad.GetTrainingGraph (null, (gradientNode, inferenceNode, inferenceSource, gradientSource) => {
                //Console.WriteLine ($"gradientNode={gradientNode}, inferenceNode={inferenceNode}, inferenceSource={inferenceSource}, gradientSource={gradientSource}");
                gradientNode.ResultImage.Format = MPSImageFeatureChannelFormat.Float32;
                if (inferenceNode is MPSNNForwardLossNode ln) {
                    lossNodesIndex[ln.Handle] = ln;
                }
                else if (inferenceNode.ResultImage.MPSHandle is LayerHandle lh &&
                         lh.Layer.GetMetalConvDataSource (device) is ConvDataSource cw) {
                    var train = trainable[lh.Layer];
                    convWeightsL.Add ((cw, train));
                    
                    //Console.WriteLine (lh);
                }                
            });

            cweights = convWeightsL.ToArray ();

            var lossNodes = lossNodesIndex.Values.ToArray ();
            if (lossNodes.Length < 1) {
                throw new InvalidOperationException ("Loss is required in order to train");
            }

            var trainingGraphTerminiImageNodes = trainingGraphTermini.Select (x => x.ResultImage).ToArray ();
            var resultsNeeded = trainingGraphTerminiImageNodes.Select (x => true).ToArray ();

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
            var trainingGraph = MPSNNGraph.Create (device, trainingGraphTerminiImageNodes, resultsNeeded);
            trainingGraph.Format = MPSImageFeatureChannelFormat.Float32;

            return trainingGraph;
        }

        public TrainingHistory Train (DataSet dataSet, float learningRate, int batchSize, int numBatches, int validateInterval)
        {
            //
            // Set the learning rate
            //
            foreach (var c in convWeights) {
                c.Weights.SetOptimizationOptions (c.Trainable, learningRate);
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

            var semaphore = new Semaphore (2, 2);

            MPSCommandBuffer? lcb = null;
            for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {
                lcb = BeginBatch (batchIndex, dataSet, batchSize, AddHistory, stopwatch, semaphore, q);

                if (batchIndex % validateInterval == 0) {
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
