using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using Metal;
using MetalTensors.Layers;
using MetalTensors.Tensors;

namespace MetalTensors
{
    public delegate IEnumerable<Tensor> LoadBatch (TensorHandle[] handles);

    public class Model
    {
        public const float DefaultLearningRate = 0.001f;
        public const int DefaultBatchSize = 32;
        public const int DefaultNumBatches = 100;
        public const int DefaultValidationInterval = 10;

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

        readonly ConcurrentDictionary<IntPtr, (InferenceGraph Inference, EvaluationGraph Evaluation, TrainingGraph Training)> graphs =
            new ConcurrentDictionary<IntPtr, (InferenceGraph Inference, EvaluationGraph Evaluation, TrainingGraph Training)> ();

        public Tensor Output => Outputs[0];
        public Tensor Input => Inputs[0];

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

        const LossType DefaultLossType = LossType.MeanSquaredError;

        public TrainingHistory Train (LoadBatch trainingData, float learningRate = DefaultLearningRate, int batchSize = DefaultBatchSize, int numBatches = DefaultNumBatches, int validationInterval = DefaultValidationInterval, IMTLDevice? device = null)
        {
            var d = device.Current ();
            var g = GetGraphs (d).Training;
            return g.Train (trainingData, learningRate, batchSize, numBatches, validationInterval);
        }

        public TrainingHistory Predict (Tensor input, IMTLDevice? device = null)
        {
            var d = device.Current ();
            var g = GetGraphs (d).Inference;
            var batchSize = 1;
            var numBatches = 1;

            return g.Predict (LoadValue, batchSize, numBatches);

            IEnumerable<Tensor> LoadValue (TensorHandle[] _)
            {
                return new[] { input };
            }
        }

        (InferenceGraph Inference, EvaluationGraph Evaluation, TrainingGraph Training) GetGraphs (IMTLDevice d)
        {
            var key = d.Handle;
            if (graphs.TryGetValue (key, out var gs))
                return gs;
            gs = CreateGraphs (d);
            if (graphs.TryAdd (key, gs))
                return gs;
            return graphs[key];
        }

        (InferenceGraph, EvaluationGraph, TrainingGraph) CreateGraphs (IMTLDevice d)
        {
            var (flatModel, trainable) = Flatten ();

            //
            // Auto add loss layer
            //
            Model trainingModel;
            if (flatModel.Layers.OfType<LossLayer> ().Any ()) {
                trainingModel = flatModel;
            }
            else {
                var labels = flatModel.Outputs.Select ((x, i) => Tensor.Labels (x.Label + " " + Tensor.DefaultLabelsLabel, x.Shape)).ToArray ();
                var losses = flatModel.Outputs.Select ((x, i) => x.Loss (labels[i], DefaultLossType)).ToArray ();
                trainingModel = new Model (flatModel.Label, flatModel.IsTrainable, flatModel.KeepDropoutDuringInference, losses);
            }

            //
            // Merge outputs in order to auto build gradients
            //
            var trainingTensor = trainingModel.Outputs.Length == 1 ?
                trainingModel.Outputs[0] :
                Tensor.Add (trainingModel.Outputs);

            //
            // Build the graphs
            //
            var evalGraph = new EvaluationGraph (Label + " Evaluation Graph", trainingTensor, KeepDropoutDuringInference, d);
            var infGraph = new InferenceGraph (Label + " Inference Graph", evalGraph.MetalGraph);
            var trainingGraph = new TrainingGraph (Label + " Training Graph", trainingTensor, trainable, evalGraph, d);

            return (infGraph, evalGraph, trainingGraph);
        }

        public (Model, Dictionary<Layer, bool>) Flatten ()
        {
            var trainable = new Dictionary<Layer, bool> ();
            foreach (var l in Layers) {
                trainable[l] = IsTrainable;
            }

            var flatOuts = Outputs.Select (FlattenTensor).ToArray ();
            var flatModel = new Model (Label, IsTrainable, KeepDropoutDuringInference, flatOuts);

            return (flatModel, trainable);

            Tensor FlattenTensor (Tensor t)
            {
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
                    return o;
                }
                if (t is LayerTensor l) {
                    return l.MapInputs (FlattenTensor);
                }
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

        public static Model Mnist ()
        {
            var (height, width) = (28, 28);
            var image = Tensor.InputImage ("image", height, width, 1);
            var weights = WeightsInit.Uniform (-0.2f, 0.2f);
            var output =
                image
                .Conv (32, size: 5, weightsInit: weights).ReLU (a: 0).MaxPool ()
                .Conv (64, size: 5, weightsInit: weights).ReLU (a: 0).MaxPool ()
                .Dense (1024, size: 7, weightsInit: weights).ReLU (a: 0)
                .Dropout (0.5f)
                .Dense (10).SoftMax ();
            var model = output.Model ("mnist");

            return model;
        }
    }
}
