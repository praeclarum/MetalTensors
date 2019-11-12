using System;
using System.Diagnostics;
using System.Threading;
using Foundation;
using MetalPerformanceShaders;

namespace MetalTensors.Tensors
{
    public class ConstantTensor : Tensor
    {
        static int nextId = 1;

        readonly int[] shape;
        readonly Lazy<MPSNNImageNode> imageNode;

        readonly Lazy<MPSImage> constantImage;

        public override int[] Shape => shape;

        public float ConstantValue { get; }

        public ConstantTensor (float constant, params int[] shape)
        {
            ConstantValue = constant;
            ValidateShape (shape);
            this.shape = shape;
            var id = Interlocked.Increment (ref nextId);
            imageNode = new Lazy<MPSNNImageNode> (() => new MPSNNImageNode (Handle), true);
            constantImage = new Lazy<MPSImage> (CreateImage, true);
        }

        public override void Copy (Span<float> destination)
        {
            var n = ValidateCopyDestination (destination);
            var c = ConstantValue;
            for (var i = 0; i < n; i++) {
                destination[i] = c;
            }
        }

        public override MPSNNImageNode GetImageNode ()
        {
            return imageNode.Value;
        }

        public override MPSImage GetImage () => constantImage.Value;

        MPSImage CreateImage ()
        {
            var imageTensor = shape.Length switch
            {
                0 => new MPSImageTensor (1, 1, 1),
                1 => new MPSImageTensor (shape[0], 1, 1),
                2 => new MPSImageTensor (shape[0], shape[1], 1),
                3 => new MPSImageTensor (shape[0], shape[1], shape[2]),
                var l => throw new InvalidOperationException ($"Cannot get image for constant data with {l} element shape"),
            };
            var image = imageTensor.Image;
            image.Fill (ConstantValue);
#if DEBUG
            var data = imageTensor.GetData ();
            Debug.Assert (data[0] == ConstantValue);
#endif
            return image;
        }
    }
}
