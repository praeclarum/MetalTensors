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
            Assert.AreEqual (ConvPadding.Valid, t.Padding);
        }

        [Test]
        public void ConcatLayer ()
        {
            var t = Deserialize (new ConcatLayer (name: "Foo"));
            Assert.AreEqual ("Foo", t.Name);
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
    }
}
