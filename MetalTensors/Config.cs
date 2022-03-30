using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Xml;
using System.Xml.Linq;

namespace MetalTensors
{
    public class Configurable
    {
        static int nextId = 0;
        public static Configurable Null { get; } = new Configurable ();

        public int Id { get; }
        public virtual Config Config => new Config { Id = Id, ObjectType = GetType ().Name };

        public Configurable ()
        {
            Id = Interlocked.Increment (ref nextId);
        }

        public byte[] Serialize ()
        {
            using var stream = new MemoryStream ();
            SaveArchive (stream);
            return stream.ToArray ();
        }

        public void Save (string path)
        {
            using var stream = new FileStream (path, FileMode.Create, FileAccess.Write, FileShare.Read);
            Save (stream);
        }

        public void Save (Stream stream)
        {
            var config = Config;
            config.Write (stream);
        }

        public void SaveArchive (string path)
        {
            using var stream = new FileStream (path, FileMode.Create, FileAccess.Write, FileShare.Read);
            SaveArchive (stream);
        }

        public void SaveArchive (Stream stream)
        {
            using var ar = new ArchiveWriter (stream);
            ar.Write (this);
        }

        public static T DeserializeObject<T> (byte[] data) where T : Configurable
        {
            using var stream = new MemoryStream (data);
            return LoadObject<T> (stream);
        }

        public static T LoadObject<T> (Stream stream) where T : Configurable
        {
            var ar = new ArchiveReader (stream);
            return ar.Read<T> ();
        }
    }

    public class ConfigCtorAttribute : Attribute { }

    public class Config : IEnumerable
    {
        readonly Dictionary<string, object> arguments = new Dictionary<string, object> ();

        public int Id {
            get => (int)(this["id"] ?? 0);
            set => Add ("id", value);
        }

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

        IEnumerator IEnumerable.GetEnumerator () => new ReadOnlyDictionary<string, object> (arguments).GetEnumerator ();

        public byte[] Serialize ()
        {
            using var stream = new MemoryStream ();
            Write (stream);
            return stream.ToArray ();
        }

        public static T Deserialize<T> (byte[] data) where T : Configurable
        {
            using var stream = new MemoryStream (data);
            return Read<T> (stream);
        }

        public string StringValue => ToString ();

        public override string ToString ()
        {
            return Encoding.UTF8.GetString (Serialize ());
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

        public void Write (string path, HashSet<Configurable>? references = null)
        {
            using var w = new StreamWriter (path, false, Encoding.UTF8);
            Write (w, references);
        }

        public void Write (Stream stream, HashSet<Configurable>? references = null)
        {
            using var w = new StreamWriter (stream, Encoding.UTF8);
            Write (w, references);
        }

        public void Write (TextWriter w, HashSet<Configurable>? references = null)
        {
            var settings = new XmlWriterSettings {
                Indent = true,
                NewLineChars = "\n",
                OmitXmlDeclaration = true,
                Encoding = Encoding.UTF8,
            };
            using var xw = XmlWriter.Create (w, settings);
            Write (xw, references);
        }

        public void Write (XmlWriter w, HashSet<Configurable>? references = null)
        {
            WriteConfig (this, w, references ?? new HashSet<Configurable> ());
        }

        static bool ValueGoesInAttribute (object? value)
        {
            return value switch {
                null => true,
                string s => s.Length < 256 && !(s.Contains ('\r') || s.Contains ('\n') || s.Contains ('\t')),
                DateTime _ => true,
                IConvertible _ => true,
                Config _ => false,
                int[] _ => true,
                Vector3 _ => true,
                Vector4 _ => true,
                _ => false
            };
        }

        static string GetAttributeString (object? value)
        {
            return value switch {
                null => "null",
                string s => s,
                DateTime d => d.ToString ("O", CultureInfo.InvariantCulture),
                IConvertible co => co.ToString (CultureInfo.InvariantCulture),
                int[] ints => string.Join (", ", ints),
                Vector3 v3 => string.Format (CultureInfo.InvariantCulture, "{0}, {1}, {2}", v3.X, v3.Y, v3.Z),
                Vector4 v4 => string.Format (CultureInfo.InvariantCulture, "{0}, {1}, {2}, {3}", v4.X, v4.Y, v4.Z, v4.W),
                var x => x.ToString (),
            };
        }

        static void WriteValueElement (object? value, XmlWriter w, HashSet<Configurable> references)
        {
            if (value is null) {
                w.WriteElementString ("Null", "");
            }
            else if (value is Config c) {
                WriteConfig (c, w, references);
            }
            else if (value is Configurable conf) {
                WriteConfigurable (conf, w, references);
            }
            else if (value is string s) {
                w.WriteElementString ("String", s);
            }
            else if (value is IConvertible co) {
                var typeName = value.GetType ().Name;
                w.WriteElementString (typeName, co.ToString (CultureInfo.InvariantCulture));
            }
            else if (value is Array l) {
                w.WriteStartElement ("Array");
                foreach (var e in l) {
                    WriteValueElement (e, w, references);
                }
                w.WriteEndElement ();
            }
            else {
                throw new NotSupportedException ($"Cannot write {value}");
            }
        }

        static void WriteConfigurable (Configurable conf, XmlWriter w, HashSet<Configurable> references)
        {
            if (references.Contains (conf)) {
                w.WriteStartElement (conf.Config.ObjectType);
                w.WriteAttributeString ("refid", conf.Id.ToString ());
                w.WriteEndElement ();
            }
            else {
                references.Add (conf);
                WriteConfig (conf.Config, w, references);
            }
        }

        static void WriteConfig (Config c, XmlWriter w, HashSet<Configurable> references)
        {
            w.WriteStartElement (c.ObjectType);
            w.WriteAttributeString ("id", c.Id.ToString ());
            var rem = new List<KeyValuePair<string, object>> ();
            foreach (var a in c.arguments) {
                if (a.Key == "id" || a.Key == "type")
                    continue;
                if (ValueGoesInAttribute (a.Value)) {
                    w.WriteAttributeString (a.Key, GetAttributeString (a.Value));
                }
                else {
                    rem.Add (a);
                }
            }
            foreach (var a in rem) {
                w.WriteStartElement (a.Key);
                WriteValueElement (a.Value, w, references);
                w.WriteEndElement ();
            }
            w.WriteEndElement ();
        }

        public static void EnableReading<T> () where T : Configurable
        {
            ConfigurableReaders.EnableReading (typeof (T));
        }

        public static T Read<T> (string path, Dictionary<int, Configurable>? references = null) where T : Configurable
        {
            return Read<T> (XDocument.Load (path), references);
        }

        public static T Read<T> (Stream stream, Dictionary<int, Configurable>? references = null) where T : Configurable
        {
            return Read<T> (XDocument.Load (stream), references);
        }

        public static T Read<T> (TextReader reader, Dictionary<int, Configurable>? references = null) where T : Configurable
        {
            return Read<T> (XDocument.Load (reader), references);
        }

        public static T Read<T> (XDocument document, Dictionary<int, Configurable>? references = null) where T : Configurable
        {
            return Read<T> (document.Root, references);
        }

        public static T Read<T> (XElement element, Dictionary<int, Configurable>? references = null) where T : Configurable
        {
            return (T)ReadConfigurable (element, references ?? new Dictionary<int, Configurable> ());
        }

        static Configurable ReadConfigurable (XElement element, Dictionary<int, Configurable> references)
        {
            var typeName = element.Name.LocalName;
            if (element.Attribute ("refid") is XAttribute ra) {
                if (references.TryGetValue (int.Parse (ra.Value), out var c)) {
                    return c;
                }
                else {
                    throw new Exception ($"Missing definition of {typeName} with refid={ra.Value}");
                }
            }
            else {
                var id = int.Parse (element.Attribute ("id").Value);
                references[id] = Configurable.Null;
                if (ConfigurableReaders.GetReader (typeName) is ConfigurableReader reader) {
                    var c = reader.Read (element, references);
                    references[id] = c;
                    return c;
                }
                throw new NotSupportedException ($"Cannot read elements of type {typeName}");
            }
        }

        static object ReadValueElement (XElement element, Type valueType, Dictionary<int, Configurable> references)
        {
            if (typeof (Configurable).IsAssignableFrom (valueType)) {
                return ReadConfigurable (element, references);
            }
            else if (typeof (Array).IsAssignableFrom (valueType)) {
                var et = valueType.GetElementType ();
                var items = element.Elements ().Select (x => ReadValueElement (x, et, references)).ToArray ();
                var result = Array.CreateInstance (et, items.Length);
                for (var i = 0; i < items.Length; i++) {
                    result.SetValue (items[i], i);
                }
                return result;
            }
            throw new NotSupportedException ($"Cannot convert element \"{element.Name}\" to {valueType}");
        }

        static readonly char[] arraySplits = new[] { ' ', ',' };

        static object ReadValueString (string value, Type valueType)
        {
            if (valueType == typeof (string))
                return value;
            if (valueType == typeof (int))
                return int.Parse (value);
            if (valueType == typeof (float))
                return float.Parse (value, CultureInfo.InvariantCulture);
            if (valueType == typeof (bool))
                return bool.Parse (value);
            if (valueType == typeof (int[]))
                return value.Split (arraySplits, StringSplitOptions.RemoveEmptyEntries).Select (x => int.Parse (x)).ToArray ();
            if (valueType == typeof (Vector3)) {
                var parts = value.Split (arraySplits, StringSplitOptions.RemoveEmptyEntries);
                var x = float.Parse (parts[0], CultureInfo.InvariantCulture);
                var y = float.Parse (parts[1], CultureInfo.InvariantCulture);
                var z = float.Parse (parts[2], CultureInfo.InvariantCulture);
                return new Vector3 (x, y, z);
            }
            if (valueType == typeof (Vector4)) {
                var parts = value.Split (arraySplits, StringSplitOptions.RemoveEmptyEntries);
                var x = float.Parse (parts[0], CultureInfo.InvariantCulture);
                var y = float.Parse (parts[1], CultureInfo.InvariantCulture);
                var z = float.Parse (parts[2], CultureInfo.InvariantCulture);
                var w = float.Parse (parts[3], CultureInfo.InvariantCulture);
                return new Vector4 (x, y, z, w);
            }
            if (valueType == typeof (DateTime))
                return DateTime.ParseExact (value, "O", System.Globalization.CultureInfo.InvariantCulture);
            if (typeof (Enum).IsAssignableFrom (valueType)) {
                return Enum.Parse (valueType, value, ignoreCase: true);
            }
            throw new NotSupportedException ($"Cannot convert string \"{value}\" to {valueType}");
        }

        static class ConfigurableReaders
        {
            static readonly Dictionary<string, ConfigurableReader> readers = new Dictionary<string, ConfigurableReader> ();

            static ConfigurableReaders ()
            {
                var asm = typeof (Model).Assembly;
                var ctype = typeof (Configurable);
                var atypes = asm.GetTypes ();
                var ctypes =
                    atypes
                    .Where (t => ctype.IsAssignableFrom (t) && !t.IsAbstract)
                    .Select (t => new ConfigurableReader (t));
                foreach (var t in ctypes) {
                    readers[t.TypeName] = t;
                }
            }

            public static ConfigurableReader? GetReader (string typeName)
            {
                if (readers.TryGetValue (typeName, out var r))
                    return r;
                return null;
            }

            public static void EnableReading (Type type)
            {
                if (readers.TryGetValue (type.Name, out var r))
                    return;
                readers[type.Name] = new ConfigurableReader (type);
            }
        }

        class ConfigurableReader
        {
            public Type ObjectType { get; }
            public ConstructorInfo Constructor { get; }
            public ParameterInfo[] Parameters { get; }
            public Dictionary<string, int> ParameterIndex { get; } = new Dictionary<string, int> ();
            public string TypeName => ObjectType.Name;

            public ConfigurableReader (Type type)
            {
                ObjectType = type;

                var ctors = type.GetConstructors ();
                Constructor =
                    ctors.FirstOrDefault (x => x.GetCustomAttribute (typeof (ConfigCtorAttribute)) != null) ??
                    ctors.OrderByDescending (x => x.GetParameters ().Length).First ();
                Parameters = Constructor.GetParameters ();
                for (var i = 0; i < Parameters.Length; i++) {
                    ParameterIndex[Parameters[i].Name] = i;
                }
            }

            public override string ToString ()
            {
                return TypeName + " Reader";
            }

            public Configurable Read (XElement element, Dictionary<int, Configurable> references)
            {
                var arguments = new object[Parameters.Length];
                var hasArg = new bool[Parameters.Length];
                var numArgs = 0;
                foreach (var a in element.Attributes ()) {
                    var name = a.Name.LocalName;
                    if (name == "id" || name == "refid")
                        continue;
                    if (ParameterIndex.TryGetValue (name, out var pindex) && arguments[pindex] is null) {
                        arguments[pindex] = ReadValueString (a.Value, Parameters[pindex].ParameterType);
                        hasArg[pindex] = true;
                        numArgs++;
                    }
                }
                foreach (var c in element.Elements ()) {
                    var name = c.Name.LocalName;
                    if (ParameterIndex.TryGetValue (name, out var pindex) && arguments[pindex] is null) {
                        arguments[pindex] = ReadValueElement (c.Elements ().First (), Parameters[pindex].ParameterType, references);
                        hasArg[pindex] = true;
                        numArgs++;
                    }
                }
                if (numArgs != Parameters.Length) {
                    var missing = Parameters.Select ((x, i) => hasArg[i] ? null : x.Name).Where (x => x != null);
                    throw new Exception ($"{TypeName} config is missing parameters: " + string.Join (", ", missing));
                }
                var obj = (Configurable)Constructor.Invoke (arguments);
                return obj;
            }
        }
    }
}
