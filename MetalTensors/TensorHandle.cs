using System;
using System.Threading;
using Foundation;
using MetalPerformanceShaders;

namespace MetalTensors
{
    public class TensorHandle : NSObject, IMPSHandle
    {
        public int Id => Tensor.Id;
        public string Label { get; }
        public Tensor Tensor { get; }

        public TensorHandle (Tensor tensor, string? label)
        {
            Tensor = tensor;
            Label = string.IsNullOrWhiteSpace (label) ? tensor.GetType ().Name + tensor.Id : label!;
        }

        public override string ToString () => Label;

        public void EncodeTo (NSCoder encoder)
        {
            encoder.Encode (new NSString (Label), "label");
        }
    }

    public class ConstantHandle : TensorHandle
    {
        public float ConstantValue { get; }

        public ConstantHandle (Tensor tensor, string? label, float constantValue)
            : base (tensor, label)
        {
            ConstantValue = constantValue;
        }

        public override string ToString () => Label + $"={ConstantValue} (Constant)";
    }

    public class InputHandle : TensorHandle
    {
        public InputHandle (Tensor tensor, string? label)
            : base (tensor, label)
        {
        }
        public override string ToString () => Label + " (Input)";
    }

    public class LabelsHandle : TensorHandle
    {
        public Tensor OutputTensor { get; }

        public LabelsHandle (Tensor tensor, Tensor outputTensor, string? label)
            : base (tensor, label)
        {
            OutputTensor = outputTensor;
        }

        public override string ToString () => Label + $" (Labels for {OutputTensor.Label})";
    }
}
