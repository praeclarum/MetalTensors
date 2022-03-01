using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Accelerate;
using CoreGraphics;
using Foundation;
using ImageIO;
using Metal;
using MetalPerformanceShaders;
using MetalTensors.Layers;
using MetalTensors.Tensors;

namespace MetalTensors
{
    public abstract class Tensor : Configurable
    {
        readonly Lazy<TensorHandle> handle;
        public TensorHandle Handle => handle.Value;
        public string Label => Handle.Label;

        public abstract int[] Shape { get; }
        public string ShapeString => "(" + string.Join (", ", Shape) + ")";
        public int Length => Shape.Aggregate (1, (a, x) => a * x);

        public virtual Tensor[] Inputs => System.Array.Empty<Tensor> ();

        readonly Lazy<MPSNNImageNode> metalImageNode;
        public virtual MPSNNImageNode GetImageNode (MetalImageNodeContext context) => metalImageNode.Value;

        public abstract bool IsStatic { get; }
        public virtual MPSImage GetMetalImage (IMTLDevice device) => throw new NotSupportedException ($"Cannot get metal image for {GetType ().Name}");
        public virtual Task CopyToAsync (MPSImage image, IMTLCommandQueue queue) => throw new NotSupportedException ($"Cannot CopyTo metal image for {GetType ().Name}");

        public override Config Config => base.Config.Add ("name", Label);

        protected Tensor (string? label = null)
        {
            handle = new Lazy<TensorHandle> (() => CreateHandle (label), true);
            metalImageNode = new Lazy<MPSNNImageNode> (() => new MPSNNImageNode (Handle), true);
        }

        protected virtual TensorHandle CreateHandle (string? label) => new TensorHandle (this, label);

        public override string ToString () => Label + " (" + string.Join (", ", Shape) + ") {type=" + GetType ().Name + "}";

        public string Format (IMTLDevice? device = null)
        {
            var len = Length;
            var vals = new float[len];
            try {
                CopyTo (vals, device);
                return "[" + string.Join (", ", vals.Take(100)) + "]";
            }
            catch (Exception ex) {
                return $"[error: {ex.Message}]";
            }
        }

        //public virtual Tensor Clone () => this;

        public abstract void CopyTo (Span<float> destination, IMTLDevice? device = null);

        public float[] ToArray (IMTLDevice? device = null)
        {
            var r = new float[Shape.GetShapeLength ()];
            CopyTo (r, device);
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
                CopyTo (elements);

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

        public virtual MPSImage CreateUninitializedImage () => MetalHelpers.CreateUninitializedImage (Shape);

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
        public static Tensor operator + (float a, Tensor b) => b.Add (a);
        public static Tensor operator + (int a, Tensor b) => b.Add (a);

        public static Tensor operator - (Tensor a, Tensor b) => a.Subtract (b);
        public static Tensor operator - (Tensor a, float b) => a.Subtract (b);
        public static Tensor operator - (Tensor a, int b) => a.Subtract (b);
        public static Tensor operator - (float a, Tensor b) => Constant (a, b).Subtract (b);
        public static Tensor operator - (int a, Tensor b) => Constant (a, b).Subtract (b);

        public static Tensor operator * (Tensor a, Tensor b) => a.Multiply (b);
        public static Tensor operator * (Tensor a, float b) => a.Multiply (b);
        public static Tensor operator * (Tensor a, int b) => a.Multiply (b);
        public static Tensor operator * (float a, Tensor b) => b.Multiply (a);
        public static Tensor operator * (int a, Tensor b) => b.Multiply (a);

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

        public static Tensor Image (NSUrl url, IMTLDevice? device = null)
        {
            return new MPSImageTensor (url, 3, device);
        }

        public static Tensor Image (NSUrl url, int featureChannels, IMTLDevice? device = null)
        {
            return new MPSImageTensor (url, featureChannels, device);
        }

        public static Tensor Image (string path, IMTLDevice? device = null)
        {
            return new MPSImageTensor (path, 3, device);
        }

        public static Tensor Image (string path, int featureChannels, IMTLDevice? device = null)
        {
            return new MPSImageTensor (path, featureChannels, device);
        }

        public static (Tensor Left, Tensor Right) ImagePair (string path, float channelScale = 1.0f, float channelOffset = 0.0f, IMTLDevice? device = null)
        {
            return MPSImageTensor.CreatePair (path, 3, channelScale, channelOffset, device);
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

        public static Tensor Input (params int[] shape)
        {
            return new InputTensor (shape);
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
            return new AbsLayer ().Call (this);
        }

        public virtual Tensor Add (Tensor other)
        {
            if (other is ConstantTensor c)
                return Add (c.ConstantValue);
            return new AddLayer ().Call (this, other);
        }

        public virtual Tensor Add (float other)
        {
            return Linear (1.0f, other);
        }

        public virtual Tensor Add (int other)
        {
            return Linear (1.0f, other);
        }

        public Tensor Apply (Model model)
        {
            return model.GetOutput (0, this);
        }

        public virtual Tensor ArgMax ()
        {
            return new ArgMaxLayer ().Call (this);
        }

        public virtual Tensor ArgMin ()
        {
            return new ArgMinLayer ().Call (this);
        }

        public Tensor AvgPool (int size = 2, int stride = 2, ConvPadding padding = ConvPadding.Valid)
        {
            return new AvgPoolLayer (size, stride, padding).Call (this);
        }

        public Tensor AvgPool (int sizeX, int sizeY, int strideX, int strideY, ConvPadding padding)
        {
            return new AvgPoolLayer (sizeX, sizeY, strideX, strideY, padding).Call (this);
        }

        public Tensor BatchNorm (float epsilon = BatchNormLayer.DefaultEpsilon)
        {
            var inChannels = Shape[^1];
            return new BatchNormLayer (inChannels, epsilon).Call (this);
        }

        public Tensor Concat (params Tensor[] others)
        {
            if (others.Length == 0)
                return this;
            return new ConcatLayer ().Call (new[] { this }.Concat (others).ToArray ());
        }

        public Tensor Conv (int featureChannels, int size = 3, int stride = 1, ConvPadding padding = ConvPadding.Same, bool bias = true, WeightsInit? weightsInit = null, float biasInit = 0.0f)
        {
            var inChannels = Shape[^1];
            return new ConvLayer (inChannels, featureChannels, size, size, stride, stride, padding, bias, weightsInit ?? WeightsInit.Default, biasInit).Call (this);
        }

        public Tensor Conv (int featureChannels, int sizeX, int sizeY, int strideX, int strideY, ConvPadding padding = ConvPadding.Same, bool bias = true, WeightsInit? weightsInit = null, float biasInit = 0.0f)
        {
            var inChannels = Shape[^1];
            return new ConvLayer (inChannels, featureChannels, sizeX, sizeY, strideX, strideY, padding, bias, weightsInit ?? WeightsInit.Default, biasInit).Call (this);
        }

        /// <summary>
        /// When the stride in any dimension is greater than 1, the convolution transpose puts (stride - 1) zeroes in-between the source 
        /// image pixels to create an expanded image.Then a convolution is done over the expanded image to generate the output of the
        /// convolution transpose.
        /// Intermediate image size = (srcSize - 1) * Stride + 1
        /// </summary>
        public Tensor ConvTranspose (int featureChannels, int size = 3, int stride = 1, ConvPadding padding = ConvPadding.Same, bool bias = true, WeightsInit? weightsInit = null, float biasInit = 0.0f)
        {
            var inChannels = Shape[^1];
            return new ConvTransposeLayer (inChannels, featureChannels, size, size, stride, stride, padding, bias, weightsInit ?? WeightsInit.Default, biasInit).Call (this);
        }

        public Tensor Dense (int featureChannels, int size = 1, bool bias = true, WeightsInit? weightsInit = null, float biasInit = 0.0f)
        {
            var inChannels = Shape[^1];
            return new DenseLayer (inChannels, featureChannels, size, size, bias, weightsInit, biasInit).Call (this);
        }

        public virtual Tensor Divide (Tensor other)
        {
            if (other is ConstantTensor c)
                return Divide (c.ConstantValue);
            return new DivideLayer ().Call (this, other);
        }

        public virtual Tensor Divide (float other)
        {
            return Linear (1.0f / other);
        }

        public virtual Tensor Divide (int other)
        {
            return Linear (1.0f / other);
        }

        public Tensor Dropout (float dropProbability)
        {
            return new DropoutLayer (dropProbability).Call (this);
        }

        public Tensor LeakyReLU (float a = ReLULayer.DefaultLeakyA)
        {
            return new ReLULayer (a).Call (this);
        }

        public Tensor Linear (float scale, float offset = 0.0f)
        {
            return new LinearLayer (scale, offset).Call (this);
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
            return new MaxLayer ().Call (this);
        }

        public Tensor MaxPool (int size = 2, int stride = 2, ConvPadding padding = ConvPadding.Valid)
        {
            return new MaxPoolLayer (size, stride, padding).Call (this);
        }

        public Tensor MaxPool (int sizeX, int sizeY, int strideX, int strideY, ConvPadding padding)
        {
            return new MaxPoolLayer (sizeX, sizeY, strideX, strideY, padding).Call (this);
        }

        public virtual Tensor Mean ()
        {
            return new MeanLayer ().Call (this);
        }

        public virtual Tensor Min ()
        {
            return new MinLayer ().Call (this);
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
            if (other is ConstantTensor c)
                return Multiply (c.ConstantValue);
            return new MultiplyLayer ().Call (this, other);
        }

        public virtual Tensor Multiply (float other)
        {
            return Linear (other);
        }

        public virtual Tensor Multiply (int other)
        {
            return Linear (other);
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
            return new ReLULayer ().Call (this);
        }

        public Tensor Sigmoid ()
        {
            return new SigmoidLayer ().Call (this);
        }

        public virtual Tensor Slice (params int[] indexes)
        {
            throw new NotSupportedException ($"Cannot slice {GetType ().Name} with {indexes.Length} int indexes");
        }

        public Tensor SoftMax ()
        {
            return new SoftMaxLayer ().Call (this);
        }

        public Tensor SpatialMean ()
        {
            return new SpatialMeanLayer ().Call (this);
        }

        public virtual Tensor Subtract (Tensor other)
        {
            if (other is ConstantTensor c)
                return Subtract (c.ConstantValue);
            return new SubtractLayer ().Call (this, other);
        }

        public virtual Tensor Subtract (float other)
        {
            return Linear (1.0f, -other);
        }

        public virtual Tensor Subtract (int other)
        {
            return Linear (1.0f, -other);
        }

        public Tensor Sum ()
        {
            return new SumLayer ().Call (this);
        }

        public virtual Tensor Tanh (string? name = null)
        {
            return new TanhLayer (name).Call (this);
        }

        public Tensor Upsample (int scaleX, int scaleY)
        {
            return new UpsampleLayer (scaleX, scaleY).Call (this);
        }

        public Tensor Upsample (int scale = 2)
        {
            return Upsample (scale, scale);
        }

        public virtual CGImage ToCGImage (float channelScale = 1.0f, float channelOffset = 0.0f, IMTLDevice? device = null)
        {
            var shape = Shape;
            if (shape.Length != 3)
                throw new Exception ($"Tensors are required to have the shape [height, width, channels] to be used as images");
            var dev = device.Current ();
            var image = GetMetalImage (dev);
            if (image == null) {
                throw new Exception ($"Failed to get metal image from {Label}");
            }
            var needsTransform = MathF.Abs(channelScale-1.0f) > 1e-7f || MathF.Abs(channelOffset) > 1e-7f;
            if (needsTransform) {
                using var transformedImage = image.Linear (channelScale, channelOffset, dev);
                return transformedImage.ToCGImage ();
            }
            else {
                return image.ToCGImage ();
            }
        }

        public virtual void SaveImage (NSUrl url, float channelScale = 1.0f, float channelOffset = 0.0f, IMTLDevice? device = null)
        {
            using var cgimage = ToCGImage (channelScale, channelOffset, device);
            var contentType = url.PathExtension == "png" ? "public.png" : "public.jpeg";
            using var dest = CGImageDestination.Create (url, contentType, 1);
            if (dest == null) {
                throw new Exception ($"Failed to create image exporter");
            }
            dest.AddImage (cgimage, new CGImageDestinationOptions {
            });
            var r = dest.Close ();                
            if (!r) {
                throw new Exception ($"Failed to save image");
            }
        }
    }
}
