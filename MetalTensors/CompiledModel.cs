using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using Metal;
using MetalTensors.Layers;
using MetalTensors.Tensors;

namespace MetalTensors
{
    public class CompiledModel
    {
        public Model Model { get; }
        public Optimizer Optimizer { get; }
        public IMTLDevice Device { get; }

        public string Label => Model.Label;

        TrainingGraph trainingGraph;
        InferenceGraph infGraph;
        EvaluationGraph evalGraph;

        public TrainingGraph TrainingGraph => trainingGraph;
        public InferenceGraph InferenceGraph => infGraph;

        public CompiledModel (Model model, Loss?[] outputLosses, Optimizer optimizer, IMTLDevice device)
        {
            Model = model;
            Optimizer = optimizer;
            Device = device;

            var (flatModel, trainable) = model.Flatten ();

            //
            // Auto add loss layer
            //
            Model trainingModel;
            if (flatModel.Layers.OfType<LossLayer> ().Any ()) {
                trainingModel = flatModel;
            }
            else {
                var labels = flatModel.Outputs.Select ((x, i) => Tensor.Labels (x.Label + " " + Tensor.DefaultLabelsLabel, x.Shape)).ToArray ();
                var losses = flatModel.Outputs.Select ((x, i) => CreateAutoLoss (x, labels[i])).ToArray ();
                trainingModel = new Model (flatModel.Label, flatModel.KeepDropoutDuringInference, losses) {
                    IsTrainable = flatModel.IsTrainable,
                };
            }

            //
            // Merge outputs in order to auto build gradients
            //
            var trainingTensor = trainingModel.Outputs.Length == 1 ?
                trainingModel.Outputs[0] :
                Tensor.Sum (trainingModel.Outputs);

            //
            // Build the graphs
            //
            evalGraph = new EvaluationGraph (Label + " Evaluation Graph", trainingTensor, Model.KeepDropoutDuringInference, device);
            infGraph = new InferenceGraph (Label + " Inference Graph", evalGraph.MetalGraph);
            trainingGraph = new TrainingGraph (Label + " Training Graph", trainingTensor, trainable, evalGraph, device);

            Tensor CreateAutoLoss (Tensor input, Tensor label)
            {
                var i = input;
                var lossType = Model.DefaultLossType;
                if (input is LayerTensor lt) {
                    if (lt.Layer is SigmoidLayer) {
                        i = lt.Inputs[0];
                        lossType = LossType.SigmoidCrossEntropy;
                    }
                    else if (lt.Layer is SoftMaxLayer) {
                        i = lt.Inputs[0];
                        lossType = LossType.SoftMaxCrossEntropy;
                    }
                }
                return i.Loss (label, lossType);
            }
        }
    }
}
