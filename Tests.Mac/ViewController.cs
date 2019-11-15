using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using AppKit;
using Foundation;

namespace Tests.Mac
{
    public partial class ViewController : NSViewController
    {
        public ViewController (IntPtr handle) : base (handle)
        {
        }

        public override async void ViewDidLoad ()
        {
            base.ViewDidLoad ();

            await Task.Run (() => {
                try {

                    RunTests ();
                }
                catch (Exception ex) {
                    Console.WriteLine (ex);
                }
            });
        }

        void ShowTestResult (TestResult tr)
        {
            Console.WriteLine (tr);
            if (!tr.Success) {
                Console.WriteLine (tr.Exception);
            }
        }

        void ShowFinalResult (bool allOK)
        {
            var m = $"ALL OK? {allOK}";
            Console.WriteLine (m);

            NSWindow? w = this.View.Window;
            if (w != null) {
                w.Title = m;
                w.BackgroundColor = allOK ? NSColor.Green : NSColor.Red;
            }
        }

        void RunTests ()
        {
            var tests = FindTests ();

            var allOK = true;

            foreach (var tf in tests) {
                foreach (var t in tf.Tests) {
                    var tr = RunTest (tf, t);
                    allOK = allOK && tr.Success;
                    BeginInvokeOnMainThread (() => ShowTestResult (tr));
                }
            }

            BeginInvokeOnMainThread (() => ShowFinalResult (allOK));
        }

        private TestResult RunTest (TestFixture tf, Test t)
        {
            try {
                var iresult = t.TestMethod.Invoke (tf.TestObject, Array.Empty<object> ());
                return new TestResult (tf, t, null);
            }
            catch (Exception ex) {
                return new TestResult (tf, t, ex);
            }
        }

        class TestResult
        {
            public bool Success => Exception == null;

            public TestFixture Fixture { get; }
            public Test Test { get; }
            public Exception? Exception { get; }

            public TestResult (TestFixture fixture, Test test, Exception? exception)
            {
                Fixture = fixture;
                Test = test;
                Exception = exception;
            }

            public override string ToString ()
            {
                return $"{Fixture}.{Test} = {Success}";
            }
        }

        class TestFixture
        {
            public object TestObject { get; }
            public Test[] Tests { get; }

            public TestFixture (object testObject, Test[] tests)
            {
                TestObject = testObject;
                Tests = tests;
            }

            public override string ToString () => TestObject.ToString ();
        }

        class Test
        {
            public MethodInfo TestMethod { get; }

            public Test (MethodInfo method)
            {
                TestMethod = method;
            }

            public override string ToString () => TestMethod.ToString ();
        }

        private TestFixture[] FindTests ()
        {
            var asmPaths = new[] { Assembly.GetCallingAssembly ().Location };

            var asms = asmPaths.Select (x => Assembly.LoadFile (x));

            var testTypes = asms.SelectMany (FindAsmTests).ToArray ();

            return testTypes.Where (x => x != null).Select (x => x!).ToArray ();

            IEnumerable<TestFixture?> FindAsmTests (Assembly asm)
            {
                return asm.Modules.SelectMany (x => x.GetTypes ()).Select (GetTestType);
            }

            TestFixture? GetTestType (Type type)
            {
                var pmeths = type.GetMethods (BindingFlags.Public | BindingFlags.Instance);

                var tmeths = pmeths.Where (x => HasAttr (x, "TestAttribute")).ToArray ();
                if (tmeths.Length == 0)
                    return null;

                var testo = Activator.CreateInstance (type);
                var tests = tmeths.Select (LoadTest).ToArray ();

                return new TestFixture (testo, tests);

                Test LoadTest (MethodInfo tt)
                {
                    return new Test (tt);
                }
            }

            static bool HasAttr (ICustomAttributeProvider m, string aname)
            {
                return m.GetCustomAttributes (false).Any (x => x.GetType ().Name == aname);
            }
        }

        public override NSObject RepresentedObject {
            get {
                return base.RepresentedObject;
            }
            set {
                base.RepresentedObject = value;
                // Update the view, if already loaded.
            }
        }
    }
}

