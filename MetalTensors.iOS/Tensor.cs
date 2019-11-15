using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using Foundation;
using Metal;
using MetalPerformanceShaders;
using MetalTensors.Layers;
using MetalTensors.Tensors;

namespace MetalTensors
{
    public abstract class Tensor
    {
        public const float DefaultLearningRate = 0.0001f;

        readonly Lazy<TensorHandle> handle;
        public TensorHandle Handle => handle.Value;
        public string Label => Handle.Label;

        public abstract int[] Shape { get; }

        public virtual Tensor[] Inputs => System.Array.Empty<Tensor> ();

        readonly Lazy<MPSNNImageNode> metalImageNode;
        public virtual MPSNNImageNode GetMetalImageNode (bool training, IMTLDevice device) => metalImageNode.Value;
        public virtual MPSImage GetMetalImage (IMTLDevice device) => throw new NotSupportedException ($"Cannot get metal image for {GetType ().Name}");

        protected Tensor (string? label = null)
        {
            handle = new Lazy<TensorHandle> (() => new TensorHandle (this, label), true);
            metalImageNode = new Lazy<MPSNNImageNode> (() => new MPSNNImageNode (Handle), true);
        }

        public override string ToString () => Label + " (" + string.Join (", ", Shape) + ")";

        //public virtual Tensor Clone () => this;

        public abstract void Copy (Span<float> destination);

        public float[] ToArray ()
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

        public static Tensor Input (string label, params int[] shape)
        {
            return new InputTensor (label, shape);
        }

        public static Tensor InputImage (string label, int height, int width, int featureChannels = 3)
        {
            return new InputTensor (label, height, width, featureChannels);
        }

        public static Tensor Labels (string label, params int[] shape)
        {
            return new LabelsTensor (label, shape);
        }

        public Tensor Apply (Model model)
        {
            return new ModelTensor (model, 0, this);
        }

        public Model Model ()
        {
            return new Model (this);
        }

        public virtual Tensor MapInputs (Dictionary<Tensor, Tensor> map)
        {
            return this;
        }

        public static Tensor Constant (float constant, params int[] shape)
        {
            return new ConstantTensor (constant, shape);
        }

        public static Tensor Zeros (params int[] shape)
        {
            return new ConstantTensor (0.0f, shape);
        }

        public static Tensor Ones (params int[] shape)
        {
            return new ConstantTensor (1.0f, shape);
        }

        public static Tensor Array (params float[] array)
        {
            return new ArrayTensor (array);
        }

        public static Tensor Array (params double[] array)
        {
            return new ArrayTensor (array.Select (x => (float)x).ToArray ());
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
            NSUrl? url = string.IsNullOrEmpty (subpath) ?
                b.GetUrlForResource (name, extension) :
                b.GetUrlForResource (name, extension, subpath);
            if (url == null)
                throw new ArgumentException ("Resource not found", nameof (name));
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

        public static Tensor Add (params Tensor[] tensors)
        {
            if (tensors.Length < 1)
                throw new ArgumentException ("Must supply at least one tensor to add", nameof (tensors));
            var r = tensors[0];
            for (var i = 1; i < tensors.Length; i++) {
                r += tensors[i];
            }
            return r;
        }

        public static Tensor operator - (Tensor a, Tensor b)
        {
            return a.Subtract (b);
        }

        public Tensor Subtract (Tensor other)
        {
            return new SubtractLayer ().GetOutput (this, other);
        }

        public static Tensor operator * (Tensor a, Tensor b)
        {
            return a.Multiply (b);
        }

        public Tensor Multiply (Tensor other)
        {
            return new MultiplyLayer ().GetOutput (this, other);
        }

        public static Tensor operator / (Tensor a, Tensor b)
        {
            return a.Divide (b);
        }

        public Tensor Divide (Tensor other)
        {
            return new DivideLayer ().GetOutput (this, other);
        }

        public Tensor Conv (int featureChannels, int size = 3, int stride = 1, bool bias = true, ConvPadding padding = ConvPadding.Same)
        {
            return new ConvLayer (featureChannels, size, size, stride, stride, bias, padding).GetOutput (this);
        }

        public Tensor Conv (int featureChannels, int sizeX, int sizeY, int strideX, int strideY, bool bias = true, ConvPadding padding = ConvPadding.Same)
        {
            return new ConvLayer (featureChannels, sizeX, sizeY, strideX, strideY, bias, padding).GetOutput (this);
        }

        public Tensor Dense (int featureChannels, bool bias = true)
        {
            return new DenseLayer (featureChannels, bias).GetOutput (this);
        }

        public Tensor ReLU (float alpha = 0.2f)
        {
            return new ReLULayer (alpha).GetOutput (this);
        }

        public Tensor Tanh ()
        {
            return new TanhLayer ().GetOutput (this);
        }

        public Tensor MaxPool (int size = 2, int stride = 2)
        {
            return new MaxPoolLayer (size, stride).GetOutput (this);
        }

        public Tensor MaxPool (int sizeX, int sizeY, int strideX, int strideY)
        {
            return new MaxPoolLayer (sizeX, sizeY, strideX, strideY).GetOutput (this);
        }

        public Tensor Upsample (int scaleX, int scaleY)
        {
            return new UpsampleLayer (scaleX, scaleY).GetOutput (this);
        }

        public Tensor Upsample (int scale = 2)
        {
            return Upsample (scale, scale);
        }

        public Tensor Loss (Tensor labels, LossType lossType, MPSCnnReductionType reductionType = MPSCnnReductionType.None, Tensor? weights = null)
        {
            if (!Shape.ShapeEquals (labels.Shape)) {
                throw new ArgumentOutOfRangeException (nameof (labels), "Labels shape must match the shape of this tensor");
            }
            var layer = new LossLayer (lossType, reductionType);
            return weights != null ?
                layer.GetOutput (this, labels, weights) :
                layer.GetOutput (this, labels);
        }

        readonly ConcurrentDictionary<IntPtr, TrainingGraph> trainingGraphs = new ConcurrentDictionary<IntPtr, TrainingGraph> ();

        public TrainingHistory Train (Func<TensorHandle[], IEnumerable<Tensor>> trainingData, float learningRate = DefaultLearningRate, int batchSize = 32, int numBatches = 10, IMTLDevice? device = null)
        {
            var d = device.Current ();

            var key = d.Handle;
            if (!trainingGraphs.TryGetValue (key, out var g)) {
                g = new TrainingGraph (this, d);
                if (!trainingGraphs.TryAdd (key, g)) {
                    g = trainingGraphs[key];
                }
            }

            return g.Train (trainingData, learningRate, batchSize, numBatches);
        }

        protected int ValidateCopyDestination (Span<float> destination)
        {
            var neededLength = Shape.GetShapeLength ();
            if (neededLength > destination.Length) {
                throw new ArgumentOutOfRangeException (nameof (destination), "Tensor copy destination memory is too small");
            }
            return neededLength;
        }

        protected static MPSImage CreateUninitializedImage (int[] shape)
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
            return image;
        }

        protected static MPSImage CreateConstantImage (int[] shape, float constantValue)
        {
            var image = CreateUninitializedImage (shape);
            image.Fill (constantValue);
#if DEBUG
            var data = new MPSImageTensor (image).ToArray ();
            Debug.Assert (data[0] == constantValue);
#endif
            return image;
        }
    }
}
