using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Net.Http;
using System.Threading.Tasks;
using Foundation;
using Metal;
using MetalPerformanceShaders;
using MetalTensors.Tensors;

namespace MetalTensors.Applications
{
    public class MnistApplication : Application
    {
        public Model Classifier { get; }

        public MnistApplication ()
        {
            Classifier = CreateModel ();
        }

        public void Train (DataSet trainingData, int batchSize = 32, int epochs = 200, IMTLDevice? device = null)
        {
            var trainImageCount = trainingData.Count;

            var numBatchesPerEpoch = trainImageCount / batchSize;

            for (var epoch = 0; epoch < epochs; epoch++) {
                Console.WriteLine ("MNIST EPOCH");
                //var discHistoryFake = Discriminator.Train (dataSet.LoadData, 0.0002f, batchSize: batchSize, numBatches: numBatchesPerEpoch, device);
                var history = Classifier.Train (trainingData, 0.0002f, batchSize: batchSize, epochs: 1, device: device);
                Console.WriteLine ("MNIST HISTORY: " + history);
            }
        }

        public static Model CreateModel ()
        {
            var (height, width) = (28, 28);
            var image = Tensor.InputImage ("image", height, width, 1);
            var labels = Tensor.Labels ("labels", 1, 1, 10);
            var weights = WeightsInit.Uniform (-0.2f, 0.2f);
            var output =
                image
                .Conv (32, size: 5, weightsInit: weights).ReLU (a: 0).MaxPool ()
                .Conv (64, size: 5, weightsInit: weights).ReLU (a: 0).MaxPool ()
                .Dense (1024, size: 7, weightsInit: weights).ReLU (a: 0)
                .Dropout (0.5f)
                .Dense (10).Loss (labels, LossType.SoftMaxCrossEntropy, ReductionType.Sum);
            var model = output.Model ("mnist");

            return model;
        }

        public class MnistDataSet : DataSet
        {
            public const int ImageSize = 28;
            const int ImagesPrefixSize = 16;
            const int LabelsPrefixSize = 8;

            readonly int numImages;
            readonly byte[] imagesData;
            readonly byte[] labelsData;
            private readonly MPSImageDescriptor trainImageDesc;
            readonly Random random;

            static readonly string[] cols = { "image", "labels" };

            public override int Count => numImages;

            public override string[] Columns => cols;

            public MnistDataSet ()
            {
                random = new Random ();
                trainImageDesc = MPSImageDescriptor.GetImageDescriptor (
                    MPSImageFeatureChannelFormat.Unorm8,
                    ImageSize, ImageSize, 1,
                    1,
                    MTLTextureUsage.ShaderWrite | MTLTextureUsage.ShaderRead);
                imagesData = ReadGZip (GetCachedPath ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"));
                labelsData = ReadGZip (GetCachedPath ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"));
                numImages = labelsData.Length - LabelsPrefixSize;
            }

            public override unsafe Tensor[] GetRow (int index)
            {
                var device = MetalExtensions.Current (null);

                fixed (byte* imagesPointer = imagesData)
                fixed (byte* labelsPointer = labelsData) {

                    var randomIndex = random.Next (numImages);

                    var trainImage = new MPSImage (device, trainImageDesc);
                    var trainImagePointer = imagesPointer + ImagesPrefixSize + randomIndex * ImageSize * ImageSize;
                    trainImage.WriteBytes ((IntPtr)trainImagePointer, MPSDataLayout.HeightPerWidthPerFeatureChannels, 0);
                    var trainTensor = new MPSImageTensor (trainImage);

                    var labelPointer = labelsPointer + LabelsPrefixSize + randomIndex;
                    var labelsValues = new float[12];
                    labelsValues[*labelPointer] = 1;
                    var labelsTensor = Tensor.Array (labelsValues);

                    return new[] { trainTensor, labelsTensor };
                }
            }

            static byte[] ReadGZip (string path)
            {
                using var fs = File.OpenRead (path);
                using var gz = new GZipStream (fs, CompressionMode.Decompress);
                using var memoryStream = new MemoryStream ();
                gz.CopyTo (memoryStream);
                return memoryStream.ToArray ();
            }

            string GetCachedPath (string url)
            {
                return GetCachedPathAsync (url).Result;
            }

            async Task<string> GetCachedPathAsync (string url)
            {
                var uri = new Uri (url, UriKind.Absolute);
                var name = Path.GetFileNameWithoutExtension (uri.AbsolutePath);
                var destPath = Path.Combine (Path.GetTempPath (), name);
                if (await Task.Run (() => File.Exists (destPath)).ConfigureAwait (false))
                    return destPath;

                var tempPath = Path.GetTempFileName ();
                using (var f = File.OpenWrite (tempPath)) {
                    Console.WriteLine ($"DOWNLOADING {uri}");
                    var client = new HttpClient ();
                    using var s = await client.GetStreamAsync (uri).ConfigureAwait (false);
                    await s.CopyToAsync (f).ConfigureAwait (false);
                }

                try {
                    File.Move (tempPath, destPath);
                }
                catch (Exception ex) {
                    Console.WriteLine (ex);
                }
                return destPath;
            }
        }
    }
}
