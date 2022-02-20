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
        const string DefaultLabelsLabel = "Labels";

        public Model Model { get; }
        public Loss?[] OutputLosses { get; }
        public float[] OutputLossWeights { get; }
        public Optimizer Optimizer { get; }
        public IMTLDevice Device { get; }
        public bool ForTraining { get; }

        public string Label => Model.Name;

        readonly Tensor[] losses;
        public Tensor[] Losses => losses;

        InferenceGraph infGraph;
        TrainingGraph? trainingGraph;
        EvaluationGraph? evalGraph;

        public InferenceGraph InferenceGraph => infGraph;
        public TrainingGraph? TrainingGraph => trainingGraph;
        public EvaluationGraph? EvaluationGraph => evalGraph;

        public CompiledModel (Model model, Loss?[] outputLosses, float[] outputLossWeights, Optimizer optimizer, IMTLDevice device, bool forTraining)
        {
            if (outputLossWeights.Length != outputLosses.Length) {
                throw new ArgumentException ("Loss weights length mismatch", nameof (outputLossWeights));
            }
            if (model.Outputs.Length != outputLosses.Length) {
                throw new ArgumentException ($"The number of provided losses ({outputLosses.Length}) must match the number of outputs ({model.Outputs.Length})", nameof (outputLosses));
            }

            Model = model;
            OutputLosses = outputLosses;
            OutputLossWeights = outputLossWeights;
            Optimizer = optimizer;
            Device = device;
            ForTraining = forTraining;
            var (flatModel, trainable) = model.Flatten ();

            if (forTraining) {

                //
                // Sum all the loss layers
                //
                var labelLosses =
                    flatModel.Outputs
                    .Select ((output, i) => {
                        var l = OutputLosses[i];
                        if (l == null)
                            return null;
                        var labels = new LabelsTensor (output.Label + " " + DefaultLabelsLabel, output, output.Shape);
                        var loss = l.Call (output, labels, OutputLossWeights[i]);
                        return loss;
                    })
                    .Where (x => x != null)
                    .Cast<Tensor> ();
                var addedLosses = model.Losses;
                losses = labelLosses.Concat (addedLosses).ToArray ();
                if (losses.Length == 0)
                    throw new InvalidOperationException ("Model has no losses");

                //
                // Build the graphs
                //
                evalGraph = new EvaluationGraph (Label + " Evaluation Graph", model.Inputs, flatModel.Outputs, losses, Model.KeepDropoutDuringInference, device);
                infGraph = new InferenceGraph (Label + " Inference Graph", model.Inputs, flatModel.Outputs, device);
                trainingGraph = new TrainingGraph (Label + " Training Graph", model.Inputs, flatModel.Outputs, losses, trainable, device);
                trainingGraph.MetalGraph.ReloadFromDataSources ();
            }
            else {
                losses = Array.Empty<Tensor> ();
                infGraph = new InferenceGraph (Label + " Inference Graph", model.Inputs, flatModel.Outputs, device);
                infGraph.MetalGraph.ReloadFromDataSources ();
            }
        }
    }
}
