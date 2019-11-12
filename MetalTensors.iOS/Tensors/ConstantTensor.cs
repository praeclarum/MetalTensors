using System;
using System.Threading;
using Foundation;
using MetalPerformanceShaders;

namespace MetalTensors.Tensors
{
    public class ConstantTensor : Tensor
    {
        static int nextId = 1;

        readonly int[] shape;
        readonly ImageNodeHandle imageNodeHandle;
        readonly Lazy<MPSNNImageNode> imageNode;

        public override int[] Shape => shape;

        public float ConstantValue { get; }

        public ConstantTensor (float constant, params int[] shape)
        {
            ConstantValue = constant;
            ValidateShape (shape);
            this.shape = shape;
            var id = Interlocked.Increment (ref nextId);
            imageNodeHandle = new ImageNodeHandle ("Constant" + id);
            imageNode = new Lazy<MPSNNImageNode> (() => new MPSNNImageNode (imageNodeHandle), true);
        }

        public override void Copy (Span<float> destination)
        {
            var n = ValidateCopyDestination (destination);
            var c = ConstantValue;
            for (var i = 0; i < n; i++) {
                destination[i] = c;
            }
        }

        public override MPSNNImageNode ToImageNode ()
        {
            return imageNode.Value;
        }

        public class ImageNodeHandle : NSObject, IMPSHandle
        {
            readonly string label;

            public string Label => label;

            public ImageNodeHandle (string label)
            {
                this.label = label;
            }

            public void EncodeTo (NSCoder encoder)
            {
                throw new NotImplementedException ();
            }
        }
    }
}
