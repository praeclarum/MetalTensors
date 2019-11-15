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
}
