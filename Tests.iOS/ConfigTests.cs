using System;
using System.IO;
using MetalPerformanceShaders;
using MetalTensors;
using MetalTensors.Applications;
using MetalTensors.Tensors;
using NUnit.Framework;

using static Tests.Imaging;

namespace Tests
{
    public class ConfigTests
    {
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
