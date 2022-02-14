using System;
using System.Threading;
using Foundation;
using MetalPerformanceShaders;

namespace MetalTensors
{
    public class TensorHandle : NSObject, IMPSHandle
    {
        static int nextId = 1;

        public string Label { get; }
        public Tensor Tensor { get; }

        public TensorHandle (Tensor tensor, string? label)
        {
            var id = Interlocked.Increment (ref nextId);
            Label = string.IsNullOrWhiteSpace (label) ? tensor.GetType ().Name + id : label!;
            Tensor = tensor;
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
        public LabelsHandle (Tensor tensor, string? label)
            : base (tensor, label)
        {
        }
        public override string ToString () => Label + " (Labels)";
    }
}
