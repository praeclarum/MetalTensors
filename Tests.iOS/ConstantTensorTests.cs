using System;
using NUnit.Framework;

using MetalTensors;
using MetalTensors.Tensors;
using MetalTensors.Layers;

namespace Tests
{
    public class ConstantTensorTests
    {
        [Test]
        public void OnesNone ()
        {
            var t = Tensor.Ones ();
            Assert.AreEqual (1, t.Shape.Length);
            Assert.AreEqual (1, t.Shape[0]);
            Assert.AreEqual (1.0f, t[0]);
        }

        [Test]
        public void ZerosSingle ()
        {
            var t = Tensor.Zeros (1);
            Assert.AreEqual (1, t.Shape.Length);
            Assert.AreEqual (1, t.Shape[0]);
            Assert.AreEqual (0.0f, t[0]);
        }

        [Test]
        public void ZerosFirstOfThree ()
        {
            var t = Tensor.Zeros (3);
            Assert.AreEqual (0.0f, t[0]);
        }

        [Test]
        public void ConstAddConstIsConst ()
        {
            var x1 = Tensor.Constant (2, 3);
            var x2 = Tensor.Constant (5, 3);
            var y = x1 + x2;
            Assert.AreEqual (7.0f, y[0]);
            Assert.AreEqual (typeof(ConstantTensor), y.GetType ());
        }

        [Test]
        public void ConstAddFloatIsConst ()
        {
            var x1 = Tensor.Constant (2, 3);
            var x2 = 5.0f;
            var y = x1 + x2;
            Assert.AreEqual (7.0f, y[0]);
            Assert.AreEqual (typeof(ConstantTensor), y.GetType ());
        }

        [Test]
        public void ConstSubConstIsConst ()
        {
            var x1 = Tensor.Constant (2, 3);
            var x2 = Tensor.Constant (5, 3);
            var y = x1 - x2;
            Assert.AreEqual (-3.0f, y[0]);
            Assert.AreEqual (typeof(ConstantTensor), y.GetType ());
        }

        [Test]
        public void ConstSubFloatIsConst ()
        {
            var x1 = Tensor.Constant (2, 3);
            var x2 = 5.0f;
            var y = x1 - x2;
            Assert.AreEqual (-3.0f, y[0]);
            Assert.AreEqual (typeof(ConstantTensor), y.GetType ());
        }

        [Test]
        public void ConstMulConstIsConst ()
        {
            var x1 = Tensor.Constant (2, 3);
            var x2 = Tensor.Constant (5, 3);
            var y = x1 * x2;
            Assert.AreEqual (10.0f, y[0]);
            Assert.AreEqual (typeof(ConstantTensor), y.GetType ());
        }

        [Test]
        public void ConstMulFloatIsConst ()
        {
            var x1 = Tensor.Constant (2, 3);
            var x2 = 5.0f;
            var y = x1 * x2;
            Assert.AreEqual (10.0f, y[0]);
            Assert.AreEqual (typeof(ConstantTensor), y.GetType ());
        }

        [Test]
        public void ConstDivConstIsConst ()
        {
            var x1 = Tensor.Constant (2, 3);
            var x2 = Tensor.Constant (5, 3);
            var y = x1 / x2;
            Assert.AreEqual (2.0f/5.0f, y[0]);
            Assert.AreEqual (typeof(ConstantTensor), y.GetType ());
        }

        [Test]
        public void ConstDivFloatIsConst ()
        {
            var x1 = Tensor.Constant (2, 3);
            var x2 = 5.0f;
            var y = x1 / x2;
            Assert.AreEqual (2.0f/5.0f, y[0]);
            Assert.AreEqual (typeof(ConstantTensor), y.GetType ());
        }

        [Test]
        public void ConstMulInputIsLinear ()
        {
            var x1 = Tensor.Constant (2, 3);
            var x2 = Tensor.Input ("x2", 3);
            var y = x1 * x2;
            Assert.AreEqual (typeof (LayerTensor), y.GetType ());
            var lt = (LayerTensor)y;
            Assert.AreEqual (typeof (LinearLayer), lt.Layer.GetType ());
        }

        [Test]
        public void InputMulManifestConstIsLinear ()
        {
            var x1 = 2.0f;
            var x2 = Tensor.Input ("x2", 3);
            var y = x2 * x1;
            Assert.AreEqual (typeof (LayerTensor), y.GetType ());
            var lt = (LayerTensor)y;
            Assert.AreEqual (typeof (LinearLayer), lt.Layer.GetType ());
        }

        [Test]
        public void ManifestConstMulInputIsLinear ()
        {
            var x1 = 2.0f;
            var x2 = Tensor.Input ("x2", 3);
            var y = x1 * x2;
            Assert.AreEqual (typeof (LayerTensor), y.GetType ());
            var lt = (LayerTensor)y;
            Assert.AreEqual (typeof (LinearLayer), lt.Layer.GetType ());
        }

        [Test]
        public void InputMulConstIsLinear ()
        {
            var x1 = Tensor.Constant (2, 3);
            var x2 = Tensor.Input ("x2", 3);
            var y = x2 * x1;
            Assert.AreEqual (typeof (LayerTensor), y.GetType ());
            var lt = (LayerTensor)y;
            Assert.AreEqual (typeof (LinearLayer), lt.Layer.GetType ());
        }

        [Test]
        public void ConstAddInputIsLinear ()
        {
            var x1 = Tensor.Constant (2, 3);
            var x2 = Tensor.Input ("x2", 3);
            var y = x1 + x2;
            Assert.AreEqual (typeof (LayerTensor), y.GetType ());
            var lt = (LayerTensor)y;
            Assert.AreEqual (typeof (LinearLayer), lt.Layer.GetType ());
        }

        [Test]
        public void InputAddConstIsLinear ()
        {
            var x1 = Tensor.Constant (2, 3);
            var x2 = Tensor.Input ("x2", 3);
            var y = x2 + x1;
            Assert.AreEqual (typeof (LayerTensor), y.GetType ());
            var lt = (LayerTensor)y;
            Assert.AreEqual (typeof (LinearLayer), lt.Layer.GetType ());
        }

        [Test]
        public void ManifestConstAddInputIsLinear ()
        {
            var x1 = 2.0f;
            var x2 = Tensor.Input ("x2", 3);
            var y = x1 + x2;
            Assert.AreEqual (typeof (LayerTensor), y.GetType ());
            var lt = (LayerTensor)y;
            Assert.AreEqual (typeof (LinearLayer), lt.Layer.GetType ());
        }

        [Test]
        public void InputAddManifestConstIsLinear ()
        {
            var x1 = 2.0f;
            var x2 = Tensor.Input ("x2", 3);
            var y = x2 + x1;
            Assert.AreEqual (typeof (LayerTensor), y.GetType ());
            var lt = (LayerTensor)y;
            Assert.AreEqual (typeof (LinearLayer), lt.Layer.GetType ());
        }

        [Test]
        public void ConstSubtractInputIsLinear ()
        {
            var x1 = Tensor.Constant (2, 3);
            var x2 = Tensor.Input ("x2", 3);
            var y = x1 - x2;
            Assert.AreEqual (typeof (LayerTensor), y.GetType ());
            var lt = (LayerTensor)y;
            Assert.AreEqual (typeof (LinearLayer), lt.Layer.GetType ());
        }

        [Test]
        public void InputSubtractConstIsLinear ()
        {
            var x1 = Tensor.Constant (2, 3);
            var x2 = Tensor.Input ("x2", 3);
            var y = x2 - x1;
            Assert.AreEqual (typeof (LayerTensor), y.GetType ());
            var lt = (LayerTensor)y;
            Assert.AreEqual (typeof (LinearLayer), lt.Layer.GetType ());
        }

        [Test]
        public void ManifestConstSubtractInputIsLinear ()
        {
            var x1 = 2.0f;
            var x2 = Tensor.Input ("x2", 3);
            var y = x1 - x2;
            Assert.AreEqual (typeof (LayerTensor), y.GetType ());
            var lt = (LayerTensor)y;
            Assert.AreEqual (typeof (LinearLayer), lt.Layer.GetType ());
        }

        [Test]
        public void InputSubtractManifestConstIsLinear ()
        {
            var x1 = 2.0f;
            var x2 = Tensor.Input ("x2", 3);
            var y = x2 - x1;
            Assert.AreEqual (typeof (LayerTensor), y.GetType ());
            var lt = (LayerTensor)y;
            Assert.AreEqual (typeof (LinearLayer), lt.Layer.GetType ());
        }

        [Test]
        public void InputDivideConstIsLinear ()
        {
            var x1 = Tensor.Constant (2, 3);
            var x2 = Tensor.Input ("x2", 3);
            var y = x2 / x1;
            Assert.AreEqual (typeof (LayerTensor), y.GetType ());
            var lt = (LayerTensor)y;
            Assert.AreEqual (typeof (LinearLayer), lt.Layer.GetType ());
        }

        [Test]
        public void InputDivideManifestConstIsLinear ()
        {
            var x1 = 2.0f;
            var x2 = Tensor.Input ("x2", 3);
            var y = x2 / x1;
            Assert.AreEqual (typeof (LayerTensor), y.GetType ());
            var lt = (LayerTensor)y;
            Assert.AreEqual (typeof (LinearLayer), lt.Layer.GetType ());
        }

    }
}
