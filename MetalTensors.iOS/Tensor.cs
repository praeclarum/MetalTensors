using System;
using System.Collections.Generic;
using System.Diagnostics;
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
        public string Label => Handle.Label;

        public abstract int[] Shape { get; }

        readonly Lazy<MPSNNImageNode> metalImageNode;
        public virtual MPSNNImageNode GetMetalImageNode (bool training, IMTLDevice device) => metalImageNode.Value;
        public virtual MPSImage GetMetalImage (IMTLDevice device) => throw new NotSupportedException ($"Cannot get metal image for {GetType ().Name}");

        protected Tensor (string? label = null)
        {
            handle = new Lazy<TensorHandle> (() => new TensorHandle (this, label), true);
            metalImageNode = new Lazy<MPSNNImageNode> (() => new MPSNNImageNode (Handle), true);
        }

        public override string ToString () => Label + " (" + string.Join (", ", Shape) + ")";

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

        public virtual TrainingHistory Train (Func<TensorHandle[], IEnumerable<Tensor>> trainingData, int batchSize = 32, int numBatches = 1, IMTLDevice? device = null)
        {
            var d = device.Current ();
            var h = new List<TrainingHistory.BatchHistory> ();

            //
            // Build the training graph
            //
            var thisImageNode = GetMetalImageNode (true, d);

            var initialGrad = new MPSNNInitialGradientNode (thisImageNode);
            var lossNodesIndex = new Dictionary<IntPtr, MPSNNForwardLossNode> ();
            var trainingGraphTermini = initialGrad.GetTrainingGraph (null, (gradientNode, inferenceNode, inferenceSource, gradientSource) => {
                //Console.WriteLine ($"gradientNode={gradientNode}, inferenceNode={inferenceNode}, inferenceSource={inferenceSource}, gradientSource={gradientSource}");
                gradientNode.ResultImage.Format = MPSImageFeatureChannelFormat.Float32;
                if (inferenceNode is MPSNNForwardLossNode ln) {
                    lossNodesIndex[ln.Handle] = ln;
                }
            });

            var lossNodes = lossNodesIndex.Values.ToArray ();
            if (lossNodes.Length < 1) {
                throw new InvalidOperationException ("Loss is required in order to train");
            }

            var trainingGraphTerminiImageNodes = trainingGraphTermini.Select (x => x.ResultImage).ToArray ();
            var resultsNeeded = trainingGraphTerminiImageNodes.Select (x => true).ToArray ();

            using var trainingGraph = MPSNNGraph.Create (d, trainingGraphTerminiImageNodes, resultsNeeded);
            trainingGraph.Format = MPSImageFeatureChannelFormat.Float32;

            var sourceHandles = trainingGraph.SourceImageHandles.Select (x => (TensorHandle)x).ToArray ();
            //var resultStateHandles = trainingGraph.ResultStateHandles;
            var intermediateHandles = trainingGraph.IntermediateImageHandles.Select (x => (LayerHandle)x).ToArray ();

            //Console.WriteLine (intermediateHandles);
            //Console.WriteLine (trainingGraph.DebugDescription);

            //
            // Train
            //
            using var queue = d.CreateCommandQueue ();

            for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {

                var bh = new Dictionary<string, Tensor[]> ();

                //
                // Load the batch
                //
                var batch = GetBatch (sourceHandles, trainingData, batchSize, d);

                using var commandBuffer = queue.CommandBuffer ();
                var intermediateImagesMA = new NSMutableArray<NSArray<MPSImage>> ();
                var destinationStates = new NSMutableArray<NSArray<MPSState>> ();
                NSArray<MPSImage>? returnBatch = trainingGraph.EncodeBatch (commandBuffer, batch, System.Array.Empty<NSArray<MPSState>> (), intermediateImagesMA, null);
                var intermediateImages = intermediateImagesMA.ToArray ();

                //
                // Synchronize needed images
                //
                if (returnBatch != null) {
                    if ((int)returnBatch.Count != batchSize) {
                        throw new Exception ("Training did not return the expected number of results");
                    }
                    MPSImageBatch.Synchronize (returnBatch, commandBuffer);
                }
                foreach (var imBatch in intermediateImages) {
                    //Console.WriteLine (ims);
                    MPSImageBatch.Synchronize (imBatch, commandBuffer);
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
                        //Console.WriteLine ($"BI{batchIndex} Results handle {r.Handle}");
                        //Console.WriteLine (r.NumberOfImages);
                    }
                }

                //Console.WriteLine ($"{intermediateImages.Length} ims");
                if (intermediateImages.Length - 1 != intermediateHandles.Length) {
                    throw new ApplicationException ("Trained intermediate images without handles");
                }
                var loss = intermediateImages.Length > 0 ?
                    intermediateImages[0].Select (x => new MPSImageTensor (x)).ToArray () :
                    System.Array.Empty<Tensor> ();
                for (var imi = 1; imi < intermediateImages.Length; imi++) {
                    var key = intermediateHandles[imi - 1].Label;
                    bh[key] = intermediateImages[imi].Select (x => new MPSImageTensor (x)).ToArray ();
                }
                h.Add (new TrainingHistory.BatchHistory (loss, bh));
                //Console.WriteLine (resultsTensors);
            }

            return new TrainingHistory (h);
        }

        static NSArray<MPSImage>[] GetBatch (TensorHandle[] sourceHandles, Func<TensorHandle[], IEnumerable<Tensor>> trainingData, int batchSize, IMTLDevice device)
        {
            var nsources = sourceHandles.Length;
            var batch = new List<List<MPSImage>> (batchSize);
            for (var i = 0; i < batchSize; i++) {
                var data = trainingData (sourceHandles);
                var dataImages = data.Select (x => x.GetMetalImage (device)).ToList ();
                batch.Add (dataImages);
            }

            var sources = new NSArray<MPSImage>[nsources];
            for (var si = 0; si < nsources; si++) {
                var b = new MPSImage[batchSize];
                for (var bi = 0; bi < batchSize; bi++) {
                    b[bi] = batch[bi][si];
                }
                sources[si] = NSArray<MPSImage>.FromNSObjects (b);
            }

            return sources;
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

        protected static MPSImage CreateConstantImage (int[] shape, float constantValue)
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
            image.Fill (constantValue);
#if DEBUG
            var data = imageTensor.ToArray ();
            Debug.Assert (data[0] == constantValue);
#endif
            return image;
        }
    }
}
