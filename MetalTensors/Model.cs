using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using Metal;
using MetalPerformanceShaders;
using MetalTensors.Layers;
using MetalTensors.Tensors;

namespace MetalTensors
{
    public class Model : Layer
    {
        public const int DefaultBatchSize = 32;
        public const int DefaultNumBatches = 100;
        public const int DefaultValidationInterval = 10;
        public const int DefaultEpochs = 10;

        public bool KeepDropoutDuringInference { get; }
        public Tensor[] Outputs { get; }

        public Tensor[] Inputs { get; }

        public Tensor[] Sources { get; }
        public Tensor[] Labels { get; }
        public Tensor[] Tensors { get; }
        public Layer[] Layers { get; }
        public Model[] Submodels { get; }

        public override int MinInputCount => Inputs.Length;

        public override int[] GetOutputShape (params Tensor[] inputs) => Outputs[0].Shape;

        readonly ConcurrentDictionary<IntPtr, CompiledModel> compiledModels =
            new ConcurrentDictionary<IntPtr, CompiledModel> ();

        public Tensor Output => Outputs[0];
        public Tensor? Input => Inputs.Length > 0 ? Inputs[0] : null;

        public Model (Tensor input, Tensor output, string? label = null)
            : this (label ?? (output.Label + " Model"), keepDropoutDuringInference: false, new[] { input }, new[] { output })
        {
        }

        public Model (Tensor[] inputs, Tensor output, string? label = null)
            : this (label ?? (output.Label + " Model"), keepDropoutDuringInference: false, inputs, new[] { output })
        {
        }

        public Model (string? label, bool keepDropoutDuringInference, Tensor[] inputs, Tensor[] outputs)
            : base (label)
        {
            if (outputs == null || outputs.Length < 1)
                throw new ArgumentException ("At least one output must be given", nameof (outputs));

            KeepDropoutDuringInference = keepDropoutDuringInference;
            Inputs = inputs;
            Outputs = outputs;

            //
            // Build graph
            //
            var handledTensors = new List<Tensor> ();
            var tensorHandled = new HashSet<Tensor> ();
            var layers = new List<Layer> ();
            var submodels = new List<Model> ();
            var sourceTensors = new List<Tensor> ();
            var inputTensors = new List<Tensor> ();
            var labelsTensors = new List<Tensor> ();
            var tensors = new List<Tensor> (outputs);
            while (tensors.Count > 0) {
                var nextTensors = new List<Tensor> ();
                foreach (var t in tensors) {
                    if (!tensorHandled.Contains (t)) {
                        handledTensors.Add (t);
                        tensorHandled.Add (t);

                        var tins = t.Inputs;
                        if (tins.Length > 0) {
                            nextTensors.AddRange (tins);
                        }
                        else {
                            if (!sourceTensors.Contains (t))
                                sourceTensors.Add (t);
                        }

                        if (t is InputTensor) {
                            if (!inputTensors.Contains (t))
                                inputTensors.Add (t);
                        }
                        else if (t is LabelsTensor) {
                            if (!labelsTensors.Contains (t))
                                labelsTensors.Add (t);
                        }
                        else if (t is LayerTensor lt) {
                            if (!layers.Contains (lt.Layer))
                                layers.Add (lt.Layer);
                        }
                        else if (t is ModelTensor mt) {
                            if (!submodels.Contains (mt.BaseModel))
                                submodels.Add (mt.BaseModel);
                        }
                    }
                }
                tensors = nextTensors;
            }

            //
            // Save the results
            //
            Tensors = handledTensors.ToArray ();
            Sources = sourceTensors.ToArray ();
            Labels = labelsTensors.ToArray ();
            Layers = layers.ToArray ();
            Submodels = submodels.ToArray ();
        }

        public override string ToString () => $"{Label} {{trainable:{IsTrainable}}}";

        public Model MapInputs (Dictionary<Tensor, Tensor> map)
        {
            var ninputs = Inputs.Select (x => x.MapInputs (map)).ToArray ();
            var noutputs = Outputs.Select (x => x.MapInputs (map)).ToArray ();
            var nm = new Model (Label, KeepDropoutDuringInference, ninputs, noutputs) {
                IsTrainable = IsTrainable,
            };
            return nm;
        }

        public Model MapInputs (Func<Tensor, Tensor> map)
        {
            var ninputs = Inputs.Select (x => x.MapInputs (map)).ToArray ();
            var noutputs = Outputs.Select (x => x.MapInputs (map)).ToArray ();
            var nm = new Model (Label, KeepDropoutDuringInference, ninputs, noutputs) {
                IsTrainable = IsTrainable,
            };
            return nm;
        }

        public Model RebuildModelWithInputs (params Tensor[] inputs)
        {
            if (inputs.Length != Inputs.Length)
                throw new ArgumentOutOfRangeException ($"Model expects {Inputs.Length} inputs, {inputs.Length} provided");

            var map = new Dictionary<Tensor, Tensor> ();
            for (int i = 0; i < inputs.Length; i++) {
                map[Inputs[i]] = inputs[i];
            }

            return MapInputs (map);
        }

        public Model Apply (Model inputModel)
        {
            var inputs = inputModel.Outputs.Select ((x, i) => inputModel.GetOutput (i, inputModel.Inputs)).ToArray ();
            var outputs = Outputs.Select ((x, i) => GetOutput (i, inputs)).ToArray ();
            return new Model (Label + "(" + inputModel.Label + ")", KeepDropoutDuringInference, inputs, outputs) {
                IsTrainable = IsTrainable,
            };
        }

        public Model Apply (params Tensor[] inputs)
        {
            var outputs = Outputs.Select ((x, i) => GetOutput (i, inputs)).ToArray ();
            return new Model (Label + "(" + string.Join (", ", inputs.Select (x => x.Label)) + ")", KeepDropoutDuringInference, inputs, outputs) {
                IsTrainable = IsTrainable,
            };
        }

        public Tensor GetOutput (int outputIndex, params Tensor[] inputs)
        {
            return new ModelTensor (this, outputIndex, inputs);
        }

        public CompiledModel Compile (Loss?[] outputLosses, float[] outputLossWeights, Optimizer optimizer, IMTLDevice? device = null, bool forTraining = true)
        {
            var d = device.Current ();
            var key = d.Handle;
            var cm = new CompiledModel (this, outputLosses, outputLossWeights, optimizer, d, forTraining: forTraining);
            compiledModels[key] = cm;
            return cm;
        }

        public CompiledModel Compile (Loss?[] outputLosses, Optimizer optimizer, IMTLDevice? device = null, bool forTraining = true)
        {
            var weights = new float[outputLosses.Length];
            Array.Fill (weights, 1.0f);
            return Compile (outputLosses, weights, optimizer, device, forTraining: forTraining);
        }

        public CompiledModel Compile (IMTLDevice? device = null, bool forTraining = true) =>
            Compile (new Loss?[Outputs.Length], new AdamOptimizer (), device, forTraining: forTraining);

        public CompiledModel Compile (Optimizer optimizer, IMTLDevice? device = null, bool forTraining = true) =>
            Compile (new Loss?[Outputs.Length], optimizer, device, forTraining: forTraining);

        public CompiledModel Compile (Loss outputLoss, IMTLDevice? device = null) =>
            Compile (new[] { outputLoss }, new AdamOptimizer (), device);

        public CompiledModel Compile (Loss outputLoss, Optimizer optimizer, IMTLDevice? device = null) =>
            Compile (new[] { outputLoss }, optimizer, device);

        public CompiledModel Compile (Loss outputLoss, float learningRate, IMTLDevice? device = null) =>
            Compile (new[] { outputLoss }, new AdamOptimizer(learningRate: learningRate), device);

        public CompiledModel Compile (Func<Tensor, Tensor, Tensor> outputLoss, Optimizer optimizer, IMTLDevice? device = null) =>
            Compile (new CustomLoss(outputLoss), optimizer, device);

        public CompiledModel? TryGetCompiledModel (IMTLDevice device)
        {
            var key = device.Handle;
            if (compiledModels.TryGetValue (key, out var gs))
                return gs;
            return null;
        }

        public TrainingHistory Fit (DataSet dataSet, int batchSize = DefaultBatchSize, int epochs = DefaultEpochs, IMTLDevice? device = null)
        {            
            var batchesPerEpoch = (dataSet.Count + batchSize - 1) / batchSize;
            return Fit (dataSet, batchSize, numBatches: batchesPerEpoch * epochs, validationInterval: batchesPerEpoch, device);
        }

        public TrainingHistory Fit (DataSet dataSet, int batchSize = DefaultBatchSize, int numBatches = DefaultNumBatches, int validationInterval = DefaultValidationInterval, IMTLDevice? device = null)
        {
            if (!(TryGetCompiledModel (device.Current ()) is CompiledModel cm)) {
                throw new InvalidOperationException ($"Models must be compiled before being Fit");
            }
            if (!(cm.TrainingGraph is TrainingGraph g)) {
                throw new InvalidOperationException ($"Model must be compiled for training before being Fit");
            }
            return g.Fit (dataSet, cm.Optimizer, batchSize, numBatches, validationInterval);
        }

        public Tensor Predict (Tensor input, IMTLDevice? device = null)
        {
            if (Inputs.Length != 1)
                throw new InvalidOperationException ($"Prediction with one input requires a model with one input (model has {Inputs.Length} inputs)");
            if (!(TryGetCompiledModel (device.Current ()) is CompiledModel cm)) {
                cm = Compile (new AdamOptimizer (), device: device, forTraining: false);
            }

            var g = cm.InferenceGraph;
            var batchSize = 1;
            var numBatches = 1;

            var h = g.Predict (DataSet.Single (Inputs[0].Label, input), batchSize, numBatches);

            return h.Batches[0].Results[0];
        }

        public (Model, Dictionary<Layer, bool>) Flatten ()
        {
            var trainable = new Dictionary<Layer, bool> ();
            foreach (var l in Layers) {
                trainable[l] = IsTrainable && l.IsTrainable;
            }
            var flattened = new Dictionary<Tensor, Tensor> ();

            var flatOuts = Outputs.Select (FlattenTensor).ToArray ();
            var flatModel = new Model (Label, KeepDropoutDuringInference, Inputs, flatOuts) {
                IsTrainable = IsTrainable,
            };

            return (flatModel, trainable);

            Tensor FlattenTensor (Tensor t)
            {
                if (flattened.TryGetValue (t, out var ft))
                    return ft;
                if (t is ModelTensor m) {
                    var inst = m.BaseModel.RebuildModelWithInputs (m.ModelInputs.Select (FlattenTensor).ToArray ());
                    var o = FlattenTensor (inst.Outputs[m.OutputIndex]);
                    foreach (var layer in GetAllLayers (o)) {
                        var lt = m.BaseModel.IsTrainable && layer.IsTrainable;
                        if (trainable.TryGetValue (layer, out var et)) {
                            lt = lt || et;
                        }
                        trainable[layer] = lt;
                    }
                    flattened[t] = o;
                    return o;
                }
                if (t is LayerTensor l) {
                    var o = l.MapInputs (FlattenTensor);
                    flattened[t] = o;
                    return o;
                }
                flattened[t] = t;
                return t;
            }

            static List<Layer> GetAllLayers (Tensor t)
            {
                var r = new List<Layer> ();
                if (t is LayerTensor lt) {
                    r.Add (lt.Layer);
                }
                foreach (var i in t.Inputs) {
                    r.AddRange (GetAllLayers (i));
                }
                return r;
            }
        }

        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            throw new NotSupportedException ($"Cannot create MPS filter nodes from models directly.");
        }
    }
}
