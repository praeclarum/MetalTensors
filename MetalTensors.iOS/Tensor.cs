using System;
using Foundation;
using Metal;
using MetalPerformanceShaders;
using MetalTensors.Layers;
using MetalTensors.Tensors;

namespace MetalTensors
{
    public abstract class Tensor
    {
        readonly Lazy<TensorHandle> handle;
        public TensorHandle Handle => handle.Value;

        public abstract int[] Shape { get; }

        readonly Lazy<MPSNNImageNode> imageNode;
        public virtual MPSNNImageNode GetMetalImageNode (IMTLDevice device) => imageNode.Value;

        protected Tensor ()
        {
            handle = new Lazy<TensorHandle> (() => new TensorHandle (this), true);
            imageNode = new Lazy<MPSNNImageNode> (() => new MPSNNImageNode (Handle), true);
        }

        public abstract void Copy (Span<float> destination);

        public float[] GetData ()
        {
            var r = new float[Shape.GetShapeLength ()];
            Copy (r);
            return r;
        }

        public virtual float this[params int[] indexes] {
            get {
                var shape = Shape;

                // This is pretty slow since the whole tensor is copied
                // Hopefully derived classes overide this property.
                var len = shape.GetShapeLength ();
                Span<float> elements = len < 1024 ?
                    stackalloc float[len] :
                    new float[len];
                Copy (elements);

                var i = 0;
                var n = Math.Min (shape.Length, indexes.Length);
                for (var j = 0; j < n; j++) {
                    i *= shape[j];
                    i += indexes[j];
                }
                return elements[i];
            }
        }

        public static Tensor Constant (float constant, params int[] shape)
        {
            if (constant == 0.0f)
                return Zeros (shape);
            if (constant == 1.0f)
                return Ones (shape);
            return new ConstantTensor (constant, shape);
        }

        public static Tensor Zeros (params int[] shape)
        {
            return new ZeroTensor (shape);
        }

        public static Tensor Ones (params int[] shape)
        {
            return new OnesTensor (shape);
        }

        public static Tensor Array (float[] array)
        {
            return new ArrayTensor (array);
        }

        public static Tensor ReadImage (NSUrl url, int featureChannels = 3, IMTLDevice? device = null)
        {
            return new MPSImageTensor (url, featureChannels, device);
        }

        public static Tensor ReadImage (string path, int featureChannels = 3, IMTLDevice? device = null)
        {
            return new MPSImageTensor (path, featureChannels, device);
        }

        public static Tensor ReadImageResource (string name, string extension, string? subpath = null, int featureChannels = 3, NSBundle? bundle = null, IMTLDevice? device = null)
        {
            var b = bundle ?? NSBundle.MainBundle;
            var url = string.IsNullOrEmpty (subpath) ?
                b.GetUrlForResource (name, extension) :
                b.GetUrlForResource (name, extension, subpath);
            return new MPSImageTensor (url, featureChannels, device);
        }

        public virtual Tensor Slice (params int[] indexes)
        {
            throw new NotSupportedException ($"Cannot slice {GetType ().Name} with {indexes.Length} int indexes");
        }

        public static Tensor operator + (Tensor a, Tensor b)
        {
            return a.Add (b);
        }

        public virtual Tensor Add (Tensor other)
        {
            return new AddLayer ().GetOutput (this, other);
        }

        public static Tensor operator - (Tensor a, Tensor b)
        {
            return a.Subtract (b);
        }

        public virtual Tensor Subtract (Tensor other)
        {
            return new SubtractLayer ().GetOutput (this, other);
        }

        public static Tensor operator * (Tensor a, Tensor b)
        {
            return a.Multiply (b);
        }

        public virtual Tensor Multiply (Tensor other)
        {
            return new MultiplyLayer ().GetOutput (this, other);
        }

        public static Tensor operator / (Tensor a, Tensor b)
        {
            return a.Divide (b);
        }

        public virtual Tensor Divide (Tensor other)
        {
            return new DivideLayer ().GetOutput (this, other);
        }

        public virtual Tensor Conv (int featureChannels, int size, int stride = 1, ConvPadding padding = ConvPadding.Same)
        {
            return new ConvLayer (featureChannels, size, stride, padding).GetOutput (this);
        }

        public virtual Tensor Conv (int featureChannels, int sizeX, int sizeY, int strideX, int strideY, ConvPadding padding = ConvPadding.Same)
        {
            return new ConvLayer (featureChannels, sizeX, sizeY, strideX, strideY, padding).GetOutput (this);
        }

        public virtual Tensor Dense (int featureChannels)
        {
            return new DenseLayer (featureChannels).GetOutput (this);
        }

        public virtual Tensor ReLU (float alpha = 0.2f)
        {
            return new ReLULayer (alpha).GetOutput (this);
        }

        public virtual Tensor MaxPool (int size = 2, int stride = 2)
        {
            return new MaxPoolLayer (size, stride).GetOutput (this);
        }

        public virtual Tensor MaxPool (int sizeX, int sizeY, int strideX, int strideY)
        {
            return new MaxPoolLayer (sizeX, sizeY, strideX, strideY).GetOutput (this);
        }

        public virtual Tensor Upsample (int scaleX, int scaleY)
        {
            return new UpsampleLayer (scaleX, scaleY).GetOutput (this);
        }

        public virtual Tensor Upsample (int scale)
        {
            return Upsample (scale, scale);
        }

        public virtual Tensor Loss (Tensor labels, LossType lossType, MPSCnnReductionType reductionType = MPSCnnReductionType.None, Tensor? weights = null)
        {
            var layer = new LossLayer (lossType, reductionType);
            return weights != null ?
                layer.GetOutput (this, labels, weights) :
                layer.GetOutput (this, labels);
        }

        public virtual History Train (Func<Tensor, Tensor> trainingData, IMTLDevice? device = null)
        {
            var d = device.Current ();
            var h = new History ();

            using var graph = new MPSNNGraph (d, GetMetalImageNode (d), true) {
                Format = MPSImageFeatureChannelFormat.Float32,
            };

            return h;
        }

        public virtual MPSImage GetMetalImage ()
        {
            throw new NotSupportedException ($"Cannot get metal image for {GetType ().Name}");
        }

        public static void ValidateShape (params int[] shape)
        {
            if (shape is null) {
                throw new ArgumentNullException (nameof (shape));
            }

            for (var i = 0; i < shape.Length; i++) {
                if (shape[i] <= 0)
                    throw new ArgumentOutOfRangeException (nameof (shape), $"Shape dimension must be > 0");
            }
        }

        protected int ValidateCopyDestination (Span<float> destination)
        {
            var neededLength = Shape.GetShapeLength ();
            if (neededLength > destination.Length) {
                throw new ArgumentOutOfRangeException (nameof (destination), "Tensor copy destination memory is too small");
            }
            return neededLength;
        }
    }
}
