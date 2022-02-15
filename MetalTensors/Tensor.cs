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
        readonly Lazy<TensorHandle> handle;
        public TensorHandle Handle => handle.Value;
        public int Id => Handle.Id;
        public string Label => Handle.Label;

        public abstract int[] Shape { get; }
        public string ShapeString => "(" + string.Join (", ", Shape) + ")";

        public virtual Tensor[] Inputs => System.Array.Empty<Tensor> ();

        readonly Lazy<MPSNNImageNode> metalImageNode;
        public virtual MPSNNImageNode GetMetalImageNode (MetalImageNodeContext context) => metalImageNode.Value;

        public abstract bool IsStatic { get; }
        public virtual MPSImage GetMetalImage (IMTLDevice device) => throw new NotSupportedException ($"Cannot get metal image for {GetType ().Name}");

        protected Tensor (string? label = null)
        {
            handle = new Lazy<TensorHandle> (() => CreateHandle (label), true);
            metalImageNode = new Lazy<MPSNNImageNode> (() => new MPSNNImageNode (Handle), true);
        }

        protected virtual TensorHandle CreateHandle (string? label) => new TensorHandle (this, label);

        public override string ToString () => Label + " (" + string.Join (", ", Shape) + ") {type=" + GetType ().Name + "}";

        //public virtual Tensor Clone () => this;

        public abstract void Copy (Span<float> destination, IMTLDevice device);

        public float[] ToArray (IMTLDevice device)
        {
            var r = new float[Shape.GetShapeLength ()];
            Copy (r, device);
            return r;
        }

        public virtual float this[params int[] indexes] {
            get {
                var shape = Shape;

                // This is pretty slow since the whole tensor is copied
                // Hopefully derived classes override this property.
                var len = shape.GetShapeLength ();
                Span<float> elements = len < 1024 ?
                    stackalloc float[len] :
                    new float[len];
                Copy (elements, ((IMTLDevice?)null).Current());

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

        protected int ValidateCopyDestination (Span<float> destination)
        {
            var neededLength = Shape.GetShapeLength ();
            if (neededLength > destination.Length) {
                throw new ArgumentOutOfRangeException (nameof (destination), "Tensor copy destination memory is too small");
            }
            return neededLength;
        }

        public static Tensor operator + (Tensor a, Tensor b) => a.Add (b);
        public static Tensor operator + (Tensor a, float b) => a.Add (b);
        public static Tensor operator + (Tensor a, int b) => a.Add (b);
        public static Tensor operator + (float a, Tensor b) => Constant (a, b).Add (b);
        public static Tensor operator + (int a, Tensor b) => Constant (a, b).Add (b);

        public static Tensor operator - (Tensor a, Tensor b) => a.Subtract (b);
        public static Tensor operator - (Tensor a, float b) => a.Subtract (b);
        public static Tensor operator - (Tensor a, int b) => a.Subtract (b);
        public static Tensor operator - (float a, Tensor b) => Constant (a, b).Subtract (b);
        public static Tensor operator - (int a, Tensor b) => Constant (a, b).Subtract (b);

        public static Tensor operator * (Tensor a, Tensor b) => a.Multiply (b);
        public static Tensor operator * (Tensor a, float b) => a.Multiply (b);
        public static Tensor operator * (Tensor a, int b) => a.Multiply (b);
        public static Tensor operator * (float a, Tensor b) => Constant (a, b).Multiply (b);
        public static Tensor operator * (int a, Tensor b) => Constant (a, b).Multiply (b);

        public static Tensor operator / (Tensor a, Tensor b) => a.Divide (b);
        public static Tensor operator / (Tensor a, float b) => a.Divide (b);
        public static Tensor operator / (Tensor a, int b) => a.Divide (b);
        public static Tensor operator / (float a, Tensor b) => Constant (a, b).Divide (b);
        public static Tensor operator / (int a, Tensor b) => Constant (a, b).Divide (b);

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

        public static Tensor Constant (float constant, string name)
        {
            return new ConstantTensor (constant, new[] { 1 }, name);
        }

        public static Tensor Constant (float constant, params int[] shape)
        {
            return new ConstantTensor (constant, shape);
        }

        public static Tensor Constant (float constant, int[] shape, string name)
        {
            return new ConstantTensor (constant, shape, name);
        }

        public static Tensor Constant (float constant, Tensor mimic)
        {
            return new ConstantTensor (constant, mimic);
        }

        public static Tensor Constant (int constant, Tensor mimic)
        {
            return new ConstantTensor (constant, mimic);
        }

        public static Tensor Image (NSUrl url, int featureChannels = 3, IMTLDevice? device = null)
        {
            return new MPSImageTensor (url, featureChannels, device);
        }

        public static Tensor Image (string path, int featureChannels = 3, IMTLDevice? device = null)
        {
            return new MPSImageTensor (path, featureChannels, device);
        }

        public static Tensor ImageResource (string name, string extension, string? subpath = null, int featureChannels = 3, NSBundle? bundle = null, IMTLDevice? device = null)
        {
            var b = bundle ?? NSBundle.MainBundle;
            var url = string.IsNullOrEmpty (subpath) ?
                b.GetUrlForResource (name, extension) :
                b.GetUrlForResource (name, extension, subpath);
            if (url == null)
                throw new ArgumentException ("Resource not found", nameof (name));
            return new MPSImageTensor (url, featureChannels, device);
        }

        public static Tensor Input (string label, params int[] shape)
        {
            return new InputTensor (label, shape);
        }

        public static Tensor Input (Tensor mimic)
        {
            return new InputTensor (mimic.Label, mimic.Shape);
        }

        public static Tensor InputImage (string label, int height, int width, int featureChannels = 3)
        {
            return new InputTensor (label, height, width, featureChannels);
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

        public static Tensor Sum (params Tensor[] tensors)
        {
            if (tensors.Length < 1)
                throw new ArgumentException ("Must supply at least one tensor to add", nameof (tensors));
            var r = tensors[0];
            for (var i = 1; i < tensors.Length; i++) {
                r += tensors[i];
            }
            return r;
        }

        public static Tensor Zeros (params int[] shape)
        {
            return new ConstantTensor (0.0f, shape);
        }

        public virtual Tensor Abs ()
        {
            return new AbsLayer ().GetOutput (this);
        }

        public virtual Tensor Add (Tensor other)
        {
            return new AddLayer ().GetOutput (this, other);
        }

        public virtual Tensor Add (float other)
        {
            return new AddLayer ().GetOutput (this, Constant (other, this));
        }

        public virtual Tensor Add (int other)
        {
            return new AddLayer ().GetOutput (this, Constant (other, this));
        }

        public Tensor Apply (Model model)
        {
            return model.GetOutput (0, this);
        }

        public virtual Tensor ArgMax ()
        {
            return new ArgMaxLayer ().GetOutput (this);
        }

        public virtual Tensor ArgMin ()
        {
            return new ArgMinLayer ().GetOutput (this);
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
            return new DenseLayer (inChannels, featureChannels, size, size, bias, weightsInit, biasInit).GetOutput (this);
        }

        public virtual Tensor Divide (Tensor other)
        {
            return new DivideLayer ().GetOutput (this, other);
        }

        public virtual Tensor Divide (float other)
        {
            return new DivideLayer ().GetOutput (this, Constant (other, this));
        }

        public virtual Tensor Divide (int other)
        {
            return new DivideLayer ().GetOutput (this, Constant (other, this));
        }

        public Tensor Dropout (float keepProbability)
        {
            return new DropoutLayer (keepProbability).GetOutput (this);
        }

        public Tensor LeakyReLU (float a = ReLULayer.DefaultLeakyA)
        {
            return new ReLULayer (a).GetOutput (this);
        }

        public Tensor Loss (Tensor truth, Loss loss, float weight = 1.0f)
        {
            return loss.Call (this, truth, weight);
        }

        public Tensor Loss (Tensor truth, LossType lossType, ReductionType reductionType, float weight = 1.0f)
        {
            return Loss (truth, MetalTensors.Loss.Builtin (lossType, reductionType), weight);
        }

        public Tensor Map (Func<Tensor, Tensor> map)
        {
            return map (MapInputs (map));
        }

        public virtual Tensor MapInputs (Dictionary<Tensor, Tensor> map)
        {
            return this;
        }

        public virtual Tensor MapInputs (Func<Tensor, Tensor> map)
        {
            return this;
        }

        public virtual Tensor Max ()
        {
            return new MaxLayer ().GetOutput (this);
        }

        public Tensor MaxPool (int size = 2, int stride = 2, ConvPadding padding = ConvPadding.Valid)
        {
            return new MaxPoolLayer (size, stride, padding).GetOutput (this);
        }

        public Tensor MaxPool (int sizeX, int sizeY, int strideX, int strideY, ConvPadding padding)
        {
            return new MaxPoolLayer (sizeX, sizeY, strideX, strideY, padding).GetOutput (this);
        }

        public virtual Tensor Mean ()
        {
            return new MeanLayer ().GetOutput (this);
        }

        public virtual Tensor Min ()
        {
            return new MinLayer ().GetOutput (this);
        }

        public Model Model (Tensor input, string? name = null, bool trainable = true)
        {
            return new Model (input, this, name) {
                IsTrainable = trainable,
            };
        }

        public Model Model (Tensor input1, Tensor input2, string? name = null, bool trainable = true)
        {
            return new Model (new[] { input1, input2 }, this, name) {
                IsTrainable = trainable,
            };
        }

        public Model Model (Tensor[] inputs, string? name = null, bool trainable = true)
        {
            return new Model (inputs, this, name) {
                IsTrainable = trainable,
            };
        }

        public virtual Tensor Multiply (Tensor other)
        {
            return new MultiplyLayer ().GetOutput (this, other);
        }

        public virtual Tensor Multiply (float other)
        {
            return new MultiplyLayer ().GetOutput (this, Constant (other, this));
        }

        public virtual Tensor Multiply (int other)
        {
            return new MultiplyLayer ().GetOutput (this, Constant (other, this));
        }

        public Tensor RemoveLayers (Func<Layer, bool> predicate)
        {
            return Map (t => {
                if (t is LayerTensor lt && predicate (lt.Layer)) {
                    return Sum (lt.Inputs);
                }
                return t;
            });
        }

        public Tensor RemoveLayers<T> () where T : Layer
        {
            return RemoveLayers (l => l is T);
        }

        public Tensor ReLU ()
        {
            return new ReLULayer ().GetOutput (this);
        }

        public Tensor Sigmoid ()
        {
            return new SigmoidLayer ().GetOutput (this);
        }

        public virtual Tensor Slice (params int[] indexes)
        {
            throw new NotSupportedException ($"Cannot slice {GetType ().Name} with {indexes.Length} int indexes");
        }

        public Tensor SoftMax ()
        {
            return new SoftMaxLayer ().GetOutput (this);
        }

        public Tensor SpatialMean ()
        {
            return new SpatialMeanLayer ().GetOutput (this);
        }

        public virtual Tensor Subtract (Tensor other)
        {
            return new SubtractLayer ().GetOutput (this, other);
        }

        public virtual Tensor Subtract (float other)
        {
            return new SubtractLayer ().GetOutput (this, Constant (other, this));
        }

        public virtual Tensor Subtract (int other)
        {
            return new SubtractLayer ().GetOutput (this, Constant (other, this));
        }

        public Tensor Sum ()
        {
            return new SumLayer ().GetOutput (this);
        }

        public virtual Tensor Tanh (string? name = null)
        {
            return new TanhLayer (name).GetOutput (this);
        }

        public Tensor Upsample (int scaleX, int scaleY)
        {
            return new UpsampleLayer (scaleX, scaleY).GetOutput (this);
        }

        public Tensor Upsample (int scale = 2)
        {
            return Upsample (scale, scale);
        }
    }
}
