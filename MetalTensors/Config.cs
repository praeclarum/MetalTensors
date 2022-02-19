using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;

namespace MetalTensors
{
    public abstract class Configurable
    {
        static int nextId = 1;
        public int Id { get; }
        public virtual Config Config => new Config { ObjectType = GetType ().Name };
        public Configurable ()
        {
            Id = Interlocked.Increment (ref nextId);
        }
    }

    public class Config : IEnumerable
    {
        readonly Dictionary<string, object> arguments = new Dictionary<string, object> ();

        public string Name {
            get => (this["name"] ?? "").ToString ();
            set => Add ("name", value);
        }

        public string ObjectType {
            get => (this["type"] ?? "").ToString ();
            set => Add ("type", value);
        }

        public int Count => arguments.Count;

        public object? this[string parameter] => arguments.TryGetValue (parameter, out var v) ? v : null;

        public Config ()
        {
        }

        public string StringValue => ToString ();

        public override string ToString ()
        {
            using var s = new StringWriter ();
            Write (s);
            return s.ToString ();
        }

        public Config Add (string parameter, object argument)
        {
            arguments[parameter] = argument;
            return this;
        }

        public Config Update (Config other)
        {
            foreach (var a in other.arguments) {
                arguments[a.Key] = a.Value;
            }
            return this;
        }

        public void Write (string path)
        {
            using var w = new StreamWriter (path, false, Encoding.UTF8);
            Write (w);
        }

        public void Write (TextWriter w)
        {
            var references = new Dictionary<object, int> ();
            WriteValue (this, w, references);
        }

        static void WriteValue (object? value, TextWriter w, Dictionary<object, int> references)
        {
            if (value is null) {
                w.Write ("null");
            }
            else if (value is string s) {
                w.Write ($"\"{EscapeString (s)}\"");
            }
            else if (value is Config c) {
                w.Write ("{");
                var head = " ";
                foreach (var a in c.arguments) {
                    w.Write (head);
                    w.Write ($"\"{EscapeString (a.Key)}\"");
                    w.Write (": ");
                    WriteValue (a.Value, w, references);
                    head = ", ";
                }
                w.Write (" }");
            }
            else if (value is IList l) {
                w.Write ("[");
                var lhead = "";
                foreach (var e in l) {
                    w.Write (lhead);
                    WriteValue (e, w, references);
                    lhead = ", ";
                }
                w.Write ("]");
            }
            else if (value is IConvertible co) {
                w.Write (co.ToString (CultureInfo.InvariantCulture));
            }
            else if (value is Configurable conf) {
                if (references.TryGetValue (value, out var _)) {
                    w.Write ($"{{ \"ref\": {conf.Id} }}");
                }
                else {
                    var vc = conf.Config.Add ("id", conf.Id);
                    WriteValue (vc, w, references);
                }
            }
            else {
                w.Write ("{}");
            }                
        }

        static string EscapeString (string s)
        {
            return s.Replace ("\\", "\\\\").Replace ("\n", "\\n").Replace ("\r", "\\r").Replace ("\t", "\\r");
        }

        IEnumerator IEnumerable.GetEnumerator ()
        {
            return new ReadOnlyDictionary<string, object> (arguments).GetEnumerator ();
        }
    }
}
