using System;
using System.Collections.Generic;
using MetalTensors.Tensors;

namespace MetalTensors
{
    public class Model
    {
        public Tensor[] Outputs { get; }

        public Tensor[] Sources { get; }
        public Tensor[] Inputs { get; }
        public Tensor[] Labels { get; }
        public Tensor[] Tensors { get; }
        public Layer[] Layers { get; }

        public Tensor TrainingTensor { get; }

        public Model (params Tensor[] outputs)
        {
            if (outputs == null || outputs.Length < 1)
                throw new ArgumentException ("At least one output must be given", nameof (outputs));

            Outputs = outputs;
            TrainingTensor = outputs.Length == 1 ?
                outputs[0] :
                Tensor.Add (outputs);

            //
            // Build graph
            //
            var handledTensors = new List<Tensor> ();
            var tensorHandled = new HashSet<Tensor> ();
            var layers = new List<Layer> ();
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
        }
    }
}
