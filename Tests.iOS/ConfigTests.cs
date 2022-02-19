using System;
using System.IO;
using MetalPerformanceShaders;
using MetalTensors;
using MetalTensors.Applications;
using MetalTensors.Layers;
using MetalTensors.Tensors;
using NUnit.Framework;

using static Tests.Imaging;

namespace Tests
{
    public class ConfigTests
    {
        T Deserialize<T> (T value) where T : Configurable
        {
            var data = value.Config.Serialize ();
            var dataString = value.Config.StringValue;
            return Config.Deserialize<T> (data);
        }

        [Test]
        public void AbsLayer ()
        {
            var t = Deserialize (new AbsLayer (name: "Foo"));
            Assert.AreEqual ("Foo", t.Name);
        }

        [Test]
        public void AddLayer ()
        {
            var t = Deserialize (new AddLayer (name: "Foo"));
            Assert.AreEqual ("Foo", t.Name);
        }

        [Test]
        public void AvgPoolLayer ()
        {
            var t = Deserialize (new AvgPoolLayer (2, 3, 5, 7, ConvPadding.Valid, name: "Foo"));
            Assert.AreEqual ("Foo", t.Name);
            Assert.AreEqual (2, t.SizeX);
            Assert.AreEqual (3, t.SizeY);
            Assert.AreEqual (5, t.StrideX);
            Assert.AreEqual (7, t.StrideY);
            Assert.AreEqual (ConvPadding.Valid, t.Padding);
        }

        [Test]
        public void BatchNormLayer ()
        {
            var t = Deserialize (new BatchNormLayer (11, epsilon: 0.0123f, name: "Foo"));
            Assert.AreEqual ("Foo", t.Name);
            Assert.AreEqual (11, t.FeatureChannels);
            Assert.AreEqual (0.0123f, t.Epsilon);
        }

        [Test]
        public void ConcatLayer ()
        {
            var t = Deserialize (new ConcatLayer (name: "Foo"));
            Assert.AreEqual ("Foo", t.Name);
        }

        [Test]
        public void ConvLayer ()
        {
            var t = Deserialize (new ConvLayer (11, 17, 2, 3, 5, 7, ConvPadding.Full, true, WeightsInit.Normal (-0.123f, 4.567f), 8.9f, name: "Foo"));
            Assert.AreEqual ("Foo", t.Name);
            Assert.AreEqual (11, t.InFeatureChannels);
            Assert.AreEqual (17, t.OutFeatureChannels);
            Assert.AreEqual (2, t.SizeX);
            Assert.AreEqual (3, t.SizeY);
            Assert.AreEqual (5, t.StrideX);
            Assert.AreEqual (7, t.StrideY);
            Assert.AreEqual (8.9f, t.BiasInit);
            Assert.AreEqual (true, t.Bias);
            Assert.AreEqual (typeof (NormalInit), t.WeightsInit.GetType ());
            Assert.AreEqual (ConvPadding.Full, t.Padding);
        }

        [Test]
        public void DenseLayer ()
        {
            var t = Deserialize (new DenseLayer (11, 17, 2, 3, true, WeightsInit.Uniform (-0.123f, 4.567f), 8.9f, name: "Foo"));
            Assert.AreEqual ("Foo", t.Name);
            Assert.AreEqual (11, t.InFeatureChannels);
            Assert.AreEqual (17, t.OutFeatureChannels);
            Assert.AreEqual (2, t.SizeX);
            Assert.AreEqual (3, t.SizeY);
            Assert.AreEqual (1, t.StrideX);
            Assert.AreEqual (1, t.StrideY);
            Assert.AreEqual (8.9f, t.BiasInit);
            Assert.AreEqual (true, t.Bias);
            Assert.AreEqual (typeof (UniformInit), t.WeightsInit.GetType ());
            Assert.AreEqual (ConvPadding.Valid, t.Padding);
        }

        [Test]
        public void DivideLayer ()
        {
            var t = Deserialize (new DivideLayer (name: "Foo"));
            Assert.AreEqual ("Foo", t.Name);
        }

        [Test]
        public void DropoutLayer ()
        {
            var t = Deserialize (new DropoutLayer (0.123f, name: "Foo"));
            Assert.AreEqual ("Foo", t.Name);
            Assert.AreEqual (0.123f, t.DropProbability);
        }

        [Test]
        public void LinearLayer ()
        {
            var t = Deserialize (new LinearLayer (0.123f, -3.14f, name: "Foo"));
            Assert.AreEqual ("Foo", t.Name);
            Assert.AreEqual (0.123f, t.Scale);
            Assert.AreEqual (-3.14f, t.Offset);
        }

        [Test]
        public void LossLayer ()
        {
            var t = Deserialize (new LossLayer (LossType.SoftMaxCrossEntropy, ReductionType.Sum, 0.123f, name: "Foo"));
            Assert.AreEqual ("Foo", t.Name);
            Assert.AreEqual (0.123f, t.Weight);
            Assert.AreEqual (LossType.SoftMaxCrossEntropy, t.LossType);
            Assert.AreEqual (ReductionType.Sum, t.ReductionType);
        }

        [Test]
        public void MaxPoolLayer ()
        {
            var t = Deserialize (new MaxPoolLayer (2, 3, 5, 7, ConvPadding.Full, name: "Foo"));
            Assert.AreEqual ("Foo", t.Name);
            Assert.AreEqual (ConvPadding.Full, t.Padding);
        }

        [Test]
        public void MultiplyLayer ()
        {
            var t = Deserialize (new MultiplyLayer (name: "Foo"));
            Assert.AreEqual ("Foo", t.Name);
        }

        [Test]
        public void ReLULayer ()
        {
            var t = Deserialize (new ReLULayer (0.123f, name: "Foo"));
            Assert.AreEqual ("Foo", t.Name);
            Assert.AreEqual (0.123f, t.A);
        }

        [Test]
        public void SigmoidLayer ()
        {
            var t = Deserialize (new SigmoidLayer (name: "Foo"));
            Assert.AreEqual ("Foo", t.Name);
        }

        [Test]
        public void SoftMaxLayer ()
        {
            var t = Deserialize (new SoftMaxLayer (name: "Foo"));
            Assert.AreEqual ("Foo", t.Name);
        }

        [Test]
        public void SubtractLayer ()
        {
            var t = Deserialize (new SubtractLayer (name: "Foo"));
            Assert.AreEqual ("Foo", t.Name);
        }

        [Test]
        public void TanhLayer ()
        {
            var t = Deserialize (new TanhLayer (name: "Foo"));
            Assert.AreEqual ("Foo", t.Name);
        }

        [Test]
        public void UpsampleLayer ()
        {
            var t = Deserialize (new UpsampleLayer (2, 3, name: "Foo"));
            Assert.AreEqual ("Foo", t.Name);
            Assert.AreEqual (2, t.ScaleX);
            Assert.AreEqual (3, t.ScaleY);
        }

        [Test]
        public void ConstTensor ()
        {
            var t = new ConstantTensor (24.0f, 2, 3, 5);
            var c = t.Config;
            Assert.AreEqual (24.0f, c["constant"]);
        }

        [Test]
        public void SerializeConstTensor ()
        {
            var t = new ConstantTensor (24.0f, 2, 3, 5);
            var data = t.Config.Serialize ();
            Assert.IsTrue (data.Length > 10);
        }

        [Test]
        public void DeserializeConstTensor ()
        {
            var t = new ConstantTensor (24.0f, 2, 3, 5);
            var data = t.Config.Serialize ();
            var t2 = Config.Deserialize<ConstantTensor> (data);
            Assert.AreEqual (24.0f, t2.ConstantValue);
        }

        [Test]
        public void DeserializeConstAddTensor ()
        {
            var t = new ConstantTensor (24.0f, 2, 3, 5);
            var data = (t + 11.0f).Config.Serialize ();
            var t2 = Config.Deserialize<Tensor> (data);
            Assert.AreEqual (35.0f, t2[0]);
        }

        [Test]
        public void DeserializeLeakyReLUTensor ()
        {
            var t = Tensor.Constant (-10.0f, 2, 3, 5).LeakyReLU (0.3f);
            var data = t.Config.Serialize ();
            var t2 = Config.Deserialize<Tensor> (data);
            Assert.AreEqual (-3.0f, t2[0]);
        }


        [Test]
        public void DeserializeModelIsTrainable ()
        {
            var input = Tensor.Input (2, 3, 5);
            var output = input.LeakyReLU (0.3f);
            var model = output.Model (input);
            Assert.AreEqual (true, model.IsTrainable);
            model.IsTrainable = false;
            Assert.AreEqual (false, model.IsTrainable);
            var data = model.Serialize ();
            var m2 = Model.Deserialize (data);
            Assert.AreEqual (false, m2.IsTrainable);
        }

        [Test]
        public void DeserializeSisoModel ()
        {
            var input = Tensor.Input (2, 3, 5);
            var output = input.LeakyReLU (0.3f);
            var model = output.Model (input);
            var testInput = Tensor.Constant (-10.0f, 2, 3, 5);
            var testOutput = model.Predict (testInput);
            var data = model.Serialize ();
            var m2 = Model.Deserialize (data);
            Assert.AreEqual (1, m2.Inputs.Length);
            Assert.AreEqual (1, m2.Outputs.Length);
            var m2Output = m2.Predict (testInput);
            Assert.AreEqual (testOutput[0], m2Output[0]);
        }

        [Test]
        public void DeserializedModelHasWeights ()
        {
            var input = Tensor.Input (2, 3, 5);
            var output = input.Conv (7, 3);
            var model = output.Model (input);
            var testInput = Tensor.Constant (1.0f, 2, 3, 5);
            var testOutput = model.Predict (testInput);
            var data = model.Serialize ();
            var m2 = Model.Deserialize (data);
            Assert.AreEqual (1, m2.Inputs.Length);
            Assert.AreEqual (1, m2.Outputs.Length);
            var m2Output = m2.Predict (testInput);
            Assert.AreEqual (testOutput[0], m2Output[0]);
        }
    }
}
