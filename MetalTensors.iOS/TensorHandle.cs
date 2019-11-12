using System;
using System.Threading;
using Foundation;
using MetalPerformanceShaders;

namespace MetalTensors
{
    public class TensorHandle : NSObject, IMPSHandle
    {
        static int nextId = 1;

        readonly string autoLabel;

        public string Label => autoLabel;

        public Tensor Tensor { get; }

        public TensorHandle (Tensor tensor)
        {
            var id = Interlocked.Increment (ref nextId);
            autoLabel = tensor.GetType ().Name + id;
            Tensor = tensor;
        }

        public override string ToString () => Label;

        public void EncodeTo (NSCoder encoder)
        {
            encoder.Encode (new NSString (autoLabel), "autoLabel");
        }
    }
}
