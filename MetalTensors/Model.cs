﻿using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using Metal;
using MetalTensors.Layers;
using MetalTensors.Tensors;

namespace MetalTensors
{
    public class Model
    {
        public const int DefaultBatchSize = 32;
        public const int DefaultNumBatches = 100;
        public const int DefaultValidationInterval = 10;
        public const int DefaultEpochs = 10;

        public string Label { get; }
        public bool IsTrainable { get; }
        public bool KeepDropoutDuringInference { get; }
        public Tensor[] Outputs { get; }

        public Tensor[] Inputs { get; }

        public Tensor[] Sources { get; }
        public Tensor[] Labels { get; }
        public Tensor[] Tensors { get; }
        public Layer[] Layers { get; }
        public Model[] Submodels { get; }

        readonly ConcurrentDictionary<IntPtr, CompiledModel> compiledModels =
            new ConcurrentDictionary<IntPtr, CompiledModel> ();

        public Tensor? Output => Outputs.Length > 0 ? Outputs[0] : null;
        public Tensor? Input => Inputs.Length > 0 ? Inputs[0] : null;

        public Model (string? label, bool trainable, bool keepDropoutDuringInference, params Tensor[] outputs)
        {
            if (outputs == null || outputs.Length < 1)
                throw new ArgumentException ("At least one output must be given", nameof (outputs));

            Label = label ?? outputs[0].Label;
            IsTrainable = trainable;
            KeepDropoutDuringInference = keepDropoutDuringInference;
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
            Inputs = inputTensors.ToArray ();
            Labels = labelsTensors.ToArray ();
            Layers = layers.ToArray ();
            Submodels = submodels.ToArray ();
        }

        public override string ToString () => $"{Label} {{trainable:{IsTrainable}}}";

        public Model Lock ()
        {
            if (!IsTrainable)
                return this;
            return new Model (Label, false, KeepDropoutDuringInference, Outputs);
        }

        public Model Unlock ()
        {
            if (IsTrainable)
                return this;
            return new Model (Label, true, KeepDropoutDuringInference, Outputs);
        }

        public Model MapInputs (Dictionary<Tensor, Tensor> map)
        {
            var noutputs = Outputs.Select (x => x.MapInputs (map)).ToArray ();
            var nm = new Model (Label, IsTrainable, KeepDropoutDuringInference, noutputs);
            return nm;
        }

        public Model MapInputs (Func<Tensor, Tensor> map)
        {
            var noutputs = Outputs.Select (x => x.MapInputs (map)).ToArray ();
            var nm = new Model (Label, IsTrainable, KeepDropoutDuringInference, noutputs);
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
            return new Model (Label + "(" + inputModel.Label + ")", IsTrainable, KeepDropoutDuringInference, outputs);
        }

        public Model Apply (params Tensor[] inputs)
        {
            var outputs = Outputs.Select ((x, i) => GetOutput (i, inputs)).ToArray ();
            return new Model (Label + "(" + string.Join (", ", inputs.Select (x => x.Label)) + ")", IsTrainable, KeepDropoutDuringInference, outputs);
        }

        public Tensor GetOutput (int outputIndex, params Tensor[] inputs)
        {
            return new ModelTensor (this, outputIndex, inputs);
        }

        public const LossType DefaultLossType = LossType.MeanSquaredError;

        public CompiledModel Compile (Optimizer optimizer, IMTLDevice? device = null)
        {
            var d = device.Current ();
            var key = d.Handle;
            var cm = new CompiledModel (this, optimizer, d);
            compiledModels[key] = cm;
            return cm;
        }

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
            var g = cm.TrainingGraph;
            return g.Fit (dataSet, cm.Optimizer, batchSize, numBatches, validationInterval);
        }

        public Tensor Predict (Tensor input, IMTLDevice? device = null)
        {
            if (Inputs.Length != 1)
                throw new InvalidOperationException ($"Prediction with one input requires a model with one input (model has {Inputs.Length} inputs)");
            if (!(TryGetCompiledModel (device.Current ()) is CompiledModel cm)) {
                cm = Compile (new AdamOptimizer (), device);
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
                trainable[l] = IsTrainable;
            }
            var flattened = new Dictionary<Tensor, Tensor> ();

            var flatOuts = Outputs.Select (FlattenTensor).ToArray ();
            var flatModel = new Model (Label, IsTrainable, KeepDropoutDuringInference, flatOuts);

            return (flatModel, trainable);

            Tensor FlattenTensor (Tensor t)
            {
                if (flattened.TryGetValue (t, out var ft))
                    return ft;
                if (t is ModelTensor m) {
                    var inst = m.BaseModel.RebuildModelWithInputs (m.ModelInputs.Select (FlattenTensor).ToArray ());
                    var o = FlattenTensor (inst.Outputs[m.OutputIndex]);
                    foreach (var layer in GetAllLayers (o)) {
                        var lt = m.BaseModel.IsTrainable;
                        if (trainable.TryGetValue (layer, out var et)) {
                            lt = lt || trainable[layer];
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
    }
}
