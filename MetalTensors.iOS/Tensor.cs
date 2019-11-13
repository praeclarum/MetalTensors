using System;
using System.Collections.Generic;
using System.Linq;
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

        readonly Lazy<MPSNNImageNode> metalImageNode;
        public virtual MPSNNImageNode GetMetalImageNode (bool training, IMTLDevice device) => metalImageNode.Value;

        protected Tensor ()
        {
            handle = new Lazy<TensorHandle> (() => new TensorHandle (this), true);
            metalImageNode = new Lazy<MPSNNImageNode> (() => new MPSNNImageNode (Handle), true);
        }

        public virtual Tensor Clone () => this;

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

        public virtual Tensor Conv (int featureChannels, int size = 3, int stride = 1, ConvPadding padding = ConvPadding.Same)
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
            if (!Shape.ShapeEquals (labels.Shape)) {
                throw new ArgumentOutOfRangeException (nameof (labels), "Labels shape must match the shape of this tensor");
            }
            var layer = new LossLayer (lossType, reductionType);
            return weights != null ?
                layer.GetOutput (this, labels, weights) :
                layer.GetOutput (this, labels);
        }

        public virtual TrainingHistory Train (Func<TensorHandle[], Tensor[]> trainingData, int batchSize = 32, int numBatches = 1, IMTLDevice? device = null)
        {
            var d = device.Current ();
            var h = new List<Tensor[]> ();

            //
            // Build the training graph
            //
            var thisImageNode = GetMetalImageNode (true, d);

            var initialGrad = new MPSNNInitialGradientNode (thisImageNode);
            var lossNodesIndex = new Dictionary<IntPtr, MPSNNForwardLossNode> ();
            var trainingGraphTermini = initialGrad.GetTrainingGraph (null, (gradientNode, inferenceNode, inferenceSource, gradientSource) => {
                Console.WriteLine ($"gradientNode={gradientNode}, inferenceNode={inferenceNode}, inferenceSource={inferenceSource}, gradientSource={gradientSource}");
                gradientNode.ResultImage.Format = MPSImageFeatureChannelFormat.Float32;
                if (inferenceNode is MPSNNForwardLossNode ln) {
                    lossNodesIndex[ln.Handle] = ln;
                }
                //Console.WriteLine (gradientNode.ResultImage.Format);
            });

            var lossNodes = lossNodesIndex.Values.ToArray ();
            if (lossNodes.Length < 1) {
                throw new InvalidOperationException ("Loss is required in order to train");
            }

            var trainingGraphTerminiImageNodes = trainingGraphTermini.Select (x => x.ResultImage).ToArray ();
            var resultsNeeded = trainingGraphTerminiImageNodes.Select (x => true).ToArray ();

            using var trainingGraph = MPSNNGraph.Create (d, trainingGraphTerminiImageNodes, resultsNeeded);
            trainingGraph.Format = MPSImageFeatureChannelFormat.Float32;
            //Console.WriteLine (trainingGraph.DebugDescription);

            //
            // Train
            //
            using var queue = d.CreateCommandQueue ();

            for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {
                using var commandBuffer = queue.CommandBuffer ();

                //
                // Load the batch
                //
                var sourceHandles = trainingGraph.SourceImageHandles.Select (x => (TensorHandle)x).ToArray ();

                var batch = GetBatch (sourceHandles, trainingData, batchSize);

                NSArray<MPSImage>? returnBatch = trainingGraph.EncodeBatch (commandBuffer, batch, System.Array.Empty<NSArray<MPSState>> ());

                //
                // Synchronize needed images
                //
                if (returnBatch != null) {
                    MPSImageBatch.Synchronize (returnBatch, commandBuffer);
                }
                foreach (var ln in lossNodes) {                    
                }

                //
                // Run the batch
                //
                commandBuffer.Commit ();
                commandBuffer.WaitUntilCompleted ();

                //
                // Process the results
                //
                if (returnBatch != null) {
                    var results = returnBatch.ToArray ();
                    foreach (var r in results) {
                        //Console.WriteLine (r.NumberOfImages);
                    }
                    var resultsTensors = results.Select (x => new MPSImageTensor (x)).ToArray ();
                    //Console.WriteLine (resultsTensors);
                }
            }

            return new TrainingHistory (h.ToArray ());
        }

        static NSArray<MPSImage>[] GetBatch (TensorHandle[] sourceHandles, Func<TensorHandle[], Tensor[]> trainingData, int batchSize)
        {
            var batch = new List<NSArray<MPSImage>> (batchSize);
            for (var i = 0; i < batchSize; i++) {
                var data = trainingData (sourceHandles);
                var sources = data.Select (x => x.GetMetalImage ()).ToArray ();
                var sourcesArray = NSArray<MPSImage>.FromNSObjects (sources);
                batch.Add (sourcesArray);
            }
            return batch.ToArray ();
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
