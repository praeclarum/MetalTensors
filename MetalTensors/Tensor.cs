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
        public const string DefaultLabelsLabel = "Labels";

        readonly Lazy<TensorHandle> handle;
        public TensorHandle Handle => handle.Value;
        public string Label => Handle.Label;

        public abstract int[] Shape { get; }
        public string ShapeString => "(" + string.Join (", ", Shape) + ")";

        public virtual Tensor[] Inputs => System.Array.Empty<Tensor> ();

        readonly Lazy<MPSNNImageNode> metalImageNode;
        public virtual MPSNNImageNode GetMetalImageNode (MetalImageNodeContext context) => metalImageNode.Value;
        public virtual MPSImage GetMetalImage (IMTLDevice device) => throw new NotSupportedException ($"Cannot get metal image for {GetType ().Name}");

        protected Tensor (string? label = null)
        {
            handle = new Lazy<TensorHandle> (() => new TensorHandle (this, label), true);
            metalImageNode = new Lazy<MPSNNImageNode> (() => new MPSNNImageNode (Handle), true);
        }

        public override string ToString () => Label + " (" + string.Join (", ", Shape) + ") {type=" + GetType ().Name + "}";

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
                var maxIndex = 1;
                for (var j = 0; j < n; j++) {
                    maxIndex *= shape[j];
                }
                for (var j = 0; j < n; j++) {
                    maxIndex /= shape[j];
                    i += indexes[j] * maxIndex;
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

        public static Tensor Labels (params int[] shape)
        {
            return new LabelsTensor (DefaultLabelsLabel, shape);
        }

        public Tensor Apply (Model model)
        {
            return model.GetOutput (0, this);
        }

        public Model Model (string? name = null, bool trainable = true, bool keepDropoutDuringInference = false)
        {
            return new Model (name, trainable, keepDropoutDuringInference, this);
        }

        public virtual Tensor MapInputs (Dictionary<Tensor, Tensor> map)
        {
            return this;
        }

        public virtual Tensor MapInputs (Func<Tensor, Tensor> map)
        {
            return this;
        }

        public Tensor Map (Func<Tensor, Tensor> map)
        {
            return map (MapInputs (map));
        }

        public Tensor RemoveLayers (Func<Layer, bool> predicate)
        {
            return Map (t => {
                if (t is LayerTensor lt && predicate (lt.Layer)) {
                    return Add (lt.Inputs);
                }
                return t;
            });
        }

        public Tensor RemoveLayers<T> () where T : Layer
        {
            return RemoveLayers (l => l is T);
        }

        public static Tensor Constant (float constant, params int[] shape)
        {
            return new ConstantTensor (constant, shape);
        }

        public static Tensor Constant (string label, float constant, params int[] shape)
        {
            return new ConstantTensor (label, constant, shape);
        }

        public static Tensor Zeros (params int[] shape)
        {
            return new ConstantTensor (0.0f, shape);
        }

        public static Tensor Ones (params int[] shape)
        {
            return new ConstantTensor (1.0f, shape);
        }

        public static Tensor OneHot (int index, int count)
        {
            var array = new float[count];
            array[index] = 1.0f;
            return new ArrayTensor (array);
        }

        public static Tensor OneHot (int index, int height, int width, int count)
        {
            var array = new float[height * width * count];
            var tindex = 0;
            for (var i = 0; i < height; i++) {
                for (var j = 0; j < width; j++) {
                    array[tindex + index] = 1;
                    tindex += count;
                }
            }
            var t = new ArrayTensor (new[] { height, width, count }, array);
            for (var i = 0; i < height; i++) {
                for (var j = 0; j < width; j++) {
                    var r = t[i, j, index];
                    if (r < 0.5) {
                        throw new Exception ("Bad one hot");
                    }
                }
            }
            return t;
        }

        public static Tensor Array (params float[] array)
        {
            return new ArrayTensor (array);
        }

        public static Tensor Array (int[] shape, params float[] array)
        {
            return new ArrayTensor (shape, array);
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

        public static Tensor operator * (float a, Tensor b)
        {
            return Constant (a, b.Shape).Multiply (b);
        }

        public static Tensor operator * (Tensor a, float b)
        {
            return a.Multiply (Constant (b, a.Shape));
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

        public Tensor AvgPool (int size = 2, int stride = 2, ConvPadding padding = ConvPadding.Valid)
        {
            return new AvgPoolLayer (size, stride, padding).GetOutput (this);
        }

        public Tensor AvgPool (int sizeX, int sizeY, int strideX, int strideY, ConvPadding padding)
        {
            return new AvgPoolLayer (sizeX, sizeY, strideX, strideY, padding).GetOutput (this);
        }

        public Tensor BatchNorm (float epsilon = BatchNormLayer.DefaultEpsilon)
        {
            var inChannels = Shape[^1];
            return new BatchNormLayer (inChannels, epsilon).GetOutput (this);
        }

        public Tensor Concat (params Tensor[] others)
        {
            if (others.Length == 0)
                return this;
            return new ConcatLayer ().GetOutput (new[] { this }.Concat (others).ToArray ());
        }

        public Tensor Conv (int featureChannels, int size = 3, int stride = 1, ConvPadding padding = ConvPadding.Same, bool bias = true, WeightsInit? weightsInit = null, float biasInit = 0.0f)
        {
            var inChannels = Shape[^1];
            return new ConvLayer (inChannels, featureChannels, size, size, stride, stride, padding, bias, weightsInit ?? WeightsInit.Default, biasInit).GetOutput (this);
        }

        public Tensor Conv (int featureChannels, int sizeX, int sizeY, int strideX, int strideY, ConvPadding padding = ConvPadding.Same, bool bias = true, WeightsInit? weightsInit = null, float biasInit = 0.0f)
        {
            var inChannels = Shape[^1];
            return new ConvLayer (inChannels, featureChannels, sizeX, sizeY, strideX, strideY, padding, bias, weightsInit ?? WeightsInit.Default, biasInit).GetOutput (this);
        }

        public Tensor ConvTranspose (int featureChannels, int size = 3, int stride = 1, ConvPadding padding = ConvPadding.Same, bool bias = true, WeightsInit? weightsInit = null, float biasInit = 0.0f)
        {
            var inChannels = Shape[^1];
            return new ConvTransposeLayer (inChannels, featureChannels, size, size, stride, stride, padding, bias, weightsInit ?? WeightsInit.Default, biasInit).GetOutput (this);
        }

        public Tensor Dense (int featureChannels, int size = 1, bool bias = true, WeightsInit? weightsInit = null, float biasInit = 0.0f)
        {
            var inChannels = Shape[^1];
            return new DenseLayer (inChannels, featureChannels, size, size, bias, weightsInit ?? WeightsInit.Default, biasInit).GetOutput (this);
        }

        public Tensor Dropout (float keepProbability)
        {
            return new DropoutLayer (keepProbability).GetOutput (this);
        }

        public Tensor ReLU (float a = 0.2f)
        {
            return new ReLULayer (a).GetOutput (this);
        }

        public Tensor Sigmoid ()
        {
            return new SigmoidLayer ().GetOutput (this);
        }

        public Tensor SoftMax ()
        {
            return new SoftMaxLayer ().GetOutput (this);
        }

        public Tensor Tanh ()
        {
            return new TanhLayer ().GetOutput (this);
        }

        public Tensor MaxPool (int size = 2, int stride = 2, ConvPadding padding = ConvPadding.Valid)
        {
            return new MaxPoolLayer (size, stride, padding).GetOutput (this);
        }

        public Tensor MaxPool (int sizeX, int sizeY, int strideX, int strideY, ConvPadding padding)
        {
            return new MaxPoolLayer (sizeX, sizeY, strideX, strideY, padding).GetOutput (this);
        }

        public Tensor Upsample (int scaleX, int scaleY)
        {
            return new UpsampleLayer (scaleX, scaleY).GetOutput (this);
        }

        public Tensor Upsample (int scale = 2)
        {
            return Upsample (scale, scale);
        }

        public Tensor Loss (Tensor labels, LossType lossType, ReductionType reductionType = ReductionType.None, Tensor? weights = null)
        {
            var layer = new LossLayer (Label + " Loss", lossType, reductionType);
            return weights != null ?
                layer.GetOutput (this, labels, weights) :
                layer.GetOutput (this, labels);
        }

        public TrainingHistory Train (DataSet dataSet, float learningRate = MetalTensors.Model.DefaultLearningRate, int batchSize = MetalTensors.Model.DefaultBatchSize, int epochs = MetalTensors.Model.DefaultEpochs, bool keepDropoutDuringInference = false, IMTLDevice? device = null)
        {
            var batchesPerEpoch = (dataSet.Count + batchSize - 1) / batchSize;
            return new Model (Label, true, keepDropoutDuringInference, this).Train (dataSet, learningRate, batchSize, batchesPerEpoch * epochs, batchesPerEpoch, device);
        }

        public TrainingHistory Train (DataSet dataSet, float learningRate = MetalTensors.Model.DefaultLearningRate, int batchSize = MetalTensors.Model.DefaultBatchSize, int numBatches = MetalTensors.Model.DefaultNumBatches, int validationInterval = MetalTensors.Model.DefaultValidationInterval, bool keepDropoutDuringInference = false, IMTLDevice? device = null)
        {
            return new Model (Label, true, keepDropoutDuringInference, this).Train (dataSet, learningRate, batchSize, numBatches, validationInterval, device);
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
                0 => new MPSImageTensor (height: 1, width: 1, featureChannels: 1),
                1 => new MPSImageTensor (height: 1, width: 1, featureChannels: shape[0]),
                2 => new MPSImageTensor (height: 1, width: shape[0], featureChannels: shape[1]),
                3 => new MPSImageTensor (height: shape[0], width: shape[1], featureChannels: shape[2]),
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
