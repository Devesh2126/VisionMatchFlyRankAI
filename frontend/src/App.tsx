import React, { useEffect, useState } from "react";
import axios from "axios";
import {
  Activity,
  BarChart3,
  CheckCircle2,
  Clock,
  Cpu,
  Database,
  History,
  Image as ImageIcon,
  Layers,
  LogOut,
  Play,
  RefreshCw,
  Search,
  Sparkles,
  Trash2,
  Upload,
} from "lucide-react";

 const API_BASE = (import.meta as any).env?.VITE_API_URL || "http://localhost:8000/api/v1";

interface UserProfile {
  id: string;
  email: string;
  role: string;
}

interface ImageItem {
  id: string;
  original_filename: string;
  mime_type: string;
  file_size_bytes: number;
  width: number;
  height: number;
  processing_status: string;
  category?: string | null;
  caption_text?: string | null;
  is_public: boolean;
  image_url: string;
  thumbnail_url: string;
  indexed_models: string[];
  created_at: string;
}

interface SearchResult {
  image_id: string;
  original_filename: string;
  category?: string | null;
  caption_text?: string | null;
  width: number;
  height: number;
  similarity_score: number;
  rank: number;
  model_key: string;
  thumbnail_url: string;
  image_url: string;
}

interface ModelInfo {
  id: string;
  model_key: string;
  model_name: string;
  model_version: string;
  hf_model_id: string;
  embedding_dimension: number;
  is_active: boolean;
  is_default: boolean;
  indexed_image_count: number;
}

interface EvaluationRun {
  id: string;
  model_key: string;
  model_name: string;
  dataset_name: string;
  num_queries: number;
  num_candidates: number;
  recall_at_1: number;
  recall_at_5: number;
  recall_at_10: number;
  precision_at_5: number;
  mean_average_precision: number;
  mean_reciprocal_rank: number;
  average_latency_ms: number;
  created_at: string;
}

interface SearchHistoryItem {
  id: string;
  query_type: string;
  query_text?: string | null;
  model_key: string;
  result_count: number;
  latency_ms: number;
  created_at: string;
}

interface DashboardStats {
  total_images: number;
  completed_jobs: number;
  failed_jobs: number;
  pending_jobs: number;
  total_embeddings: number;
  active_model_key: string;
  recent_searches: SearchHistoryItem[];
}

export default function App() {
  const [token, setToken] = useState<string | null>(
    localStorage.getItem("vm_token")
  );
  const [user, setUser] = useState<UserProfile | null>(null);
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("demo@visionmatch.ai");
  const [password, setPassword] = useState("VisionMatch123!");
  const [authError, setAuthError] = useState<string | null>(null);

  const [activeTab, setActiveTab] = useState<
    "dashboard" | "search" | "upload" | "gallery" | "match" | "models" | "history"
  >("dashboard");

  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("clip-vit-b32");
  const [images, setImages] = useState<ImageItem[]>([]);
  const [evaluations, setEvaluations] = useState<EvaluationRun[]>([]);
  const [history, setHistory] = useState<SearchHistoryItem[]>([]);

  // Search State
  const [searchMode, setSearchMode] = useState<"text" | "image">("text");
  const [queryText, setQueryText] = useState(
    "A bright red sports car parked beside a sunset ocean beach"
  );
  const [searchImageFile, setSearchImageFile] = useState<File | null>(null);
  const [topK, setTopK] = useState<number>(8);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searchMeta, setSearchMeta] = useState<{
    emb_ms: number;
    ret_ms: number;
    total_ms: number;
  } | null>(null);
  const [isSearching, setIsSearching] = useState(false);

  // Upload State
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadCategory, setUploadCategory] = useState("automotive");
  const [uploadCaption, setUploadCaption] = useState("");
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  // Cross-modal Match State
  const [matchImageId, setMatchImageId] = useState<string>("");
  const [candidateCaptions, setCandidateCaptions] = useState<string>(
    "A red sports car parked at sunset by the ocean\nSnow-capped alpine mountain peaks under blue sky\nA golden retriever dog sitting on green lawn\nFuturistic cyberpunk neon city street at night"
  );
  const [matchRankings, setMatchRankings] = useState<
    {
      description: string;
      cosine_similarity: number;
      relative_alignment_pct: number;
      rank: number;
    }[]
  >([]);

  const [isEvalRunning, setIsEvalRunning] = useState(false);

  const api = axios.create({
    baseURL: API_BASE,
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError(null);
    try {
      const endpoint = authMode === "login" ? "/auth/login" : "/auth/register";
      const res = await axios.post(`${API_BASE}${endpoint}`, {
        email,
        password,
      });
      localStorage.setItem("vm_token", res.data.access_token);
      setToken(res.data.access_token);
      setUser(res.data.user);
    } catch (err: any) {
      setAuthError(
        err?.response?.data?.detail ||
          "Authentication failed. Make sure the FastAPI backend is running on port 8000."
      );
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("vm_token");
    setToken(null);
    setUser(null);
  };

  const fetchAllData = async () => {
    if (!token) return;
    try {
      const [meRes, statsRes, modelsRes, imgRes, evalRes, histRes] =
        await Promise.all([
          api.get("/auth/me"),
          api.get("/dashboard/stats"),
          api.get("/models"),
          api.get("/images"),
          api.get("/evaluations"),
          api.get("/search/history"),
        ]);
      setUser(meRes.data);
      setStats(statsRes.data);
      setModels(modelsRes.data);
      setImages(imgRes.data);
      if (imgRes.data.length > 0 && !matchImageId) {
        setMatchImageId(imgRes.data[0].id);
      }
      setEvaluations(evalRes.data);
      setHistory(histRes.data);
    } catch (err: any) {
      if (err?.response?.status === 401) {
        handleLogout();
      }
    }
  };

  useEffect(() => {
    fetchAllData();
  }, [token]);

  const executeSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setIsSearching(true);
    try {
      if (searchMode === "text") {
        const res = await api.post("/search/text", {
          query: queryText,
          model_key: selectedModel,
          top_k: topK,
          include_public: true,
        });
        setSearchResults(res.data.results);
        setSearchMeta({
          emb_ms: res.data.embedding_latency_ms,
          ret_ms: res.data.retrieval_latency_ms,
          total_ms: res.data.total_latency_ms,
        });
      } else if (searchImageFile) {
        const form = new FormData();
        form.append("file", searchImageFile);
        form.append("model_key", selectedModel);
        form.append("top_k", String(topK));
        const res = await api.post("/search/image", form);
        setSearchResults(res.data.results);
        setSearchMeta({
          emb_ms: res.data.embedding_latency_ms,
          ret_ms: res.data.retrieval_latency_ms,
          total_ms: res.data.total_latency_ms,
        });
      }
      fetchAllData();
    } finally {
      setIsSearching(false);
    }
  };

  const findSimilarByImageId = async (imageId: string) => {
    setActiveTab("search");
    setIsSearching(true);
    try {
      const form = new FormData();
      form.append("image_id", imageId);
      form.append("model_key", selectedModel);
      form.append("top_k", String(topK));
      const res = await api.post("/search/image", form);
      setSearchResults(res.data.results);
      setSearchMeta({
        emb_ms: res.data.embedding_latency_ms,
        ret_ms: res.data.retrieval_latency_ms,
        total_ms: res.data.total_latency_ms,
      });
      fetchAllData();
    } finally {
      setIsSearching(false);
    }
  };

  const handleUploadImage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFile) return;
    setUploadStatus("Validating image, stripping EXIF, and generating embedding...");
    try {
      const form = new FormData();
      form.append("file", uploadFile);
      form.append("category", uploadCategory);
      form.append("caption_text", uploadCaption);
      form.append("model_key", selectedModel);
      form.append("is_public", "true");
      const res = await api.post("/images/upload", form);
      setUploadStatus(
        `${res.data.message} (Status: ${res.data.processing_status})`
      );
      setUploadFile(null);
      setUploadCaption("");
      setTimeout(() => fetchAllData(), 1200);
    } catch (err: any) {
      setUploadStatus(
        `Error: ${err?.response?.data?.detail || "Upload failed"}`
      );
    }
  };

  const handleRunCrossModalMatch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!matchImageId) return;
    const lines = candidateCaptions
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean);
    const res = await api.post("/search/match", {
      image_id: matchImageId,
      candidate_descriptions: lines,
      model_key: selectedModel,
    });
    setMatchRankings(res.data.rankings);
  };

  const handleRunEvaluation = async (modelKey: string) => {
    setIsEvalRunning(true);
    try {
      await api.post("/evaluations/run", {
        model_key: modelKey,
        dataset_name: "curated_multimodal_benchmark_v1",
      });
      await fetchAllData();
    } finally {
      setIsEvalRunning(false);
    }
  };

  const handleDeleteImage = async (id: string) => {
    await api.delete(`/images/${id}`);
    fetchAllData();
  };

  if (!token) {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col justify-center items-center p-6">
        <div className="max-w-md w-full bg-slate-900/90 border border-slate-800 rounded-2xl p-8 shadow-2xl">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2.5 rounded-xl bg-indigo-600/20 border border-indigo-500/40 text-indigo-400">
              <Sparkles className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">
                VisionMatch AI
              </h1>
              <p className="text-xs text-slate-400">
                Understand Images. Search by Meaning.
              </p>
            </div>
          </div>

          <div className="flex gap-2 mb-6 bg-slate-950 p-1 rounded-lg border border-slate-800">
            <button
              type="button"
              onClick={() => setAuthMode("login")}
              className={`flex-1 py-2 text-xs font-semibold rounded-md transition ${
                authMode === "login"
                  ? "bg-indigo-600 text-white"
                  : "text-slate-400"
              }`}
            >
              Sign In
            </button>
            <button
              type="button"
              onClick={() => setAuthMode("register")}
              className={`flex-1 py-2 text-xs font-semibold rounded-md transition ${
                authMode === "register"
                  ? "bg-indigo-600 text-white"
                  : "text-slate-400"
              }`}
            >
              Create Account
            </button>
          </div>

          {authError && (
            <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 text-xs">
              {authError}
            </div>
          )}

          <form onSubmit={handleAuth} className="space-y-4">
            <div>
              <label className="block text-xs text-slate-400 mb-1">
                Email Address
              </label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-3.5 py-2.5 rounded-lg bg-slate-950 border border-slate-800 text-sm focus:border-indigo-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">
                Password
              </label>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-3.5 py-2.5 rounded-lg bg-slate-950 border border-slate-800 text-sm focus:border-indigo-500 focus:outline-none"
              />
            </div>
            <button
              type="submit"
              className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold rounded-lg text-sm transition"
            >
              {authMode === "login"
                ? "Launch Platform"
                : "Register & Continue"}
            </button>
          </form>

          <div className="mt-5 pt-4 border-t border-slate-800 text-xs text-slate-400">
            Pre-seeded Demo Credentials:{" "}
            <span className="text-slate-200 font-mono">
              demo@visionmatch.ai
            </span>{" "}
            / <span className="text-slate-200 font-mono">VisionMatch123!</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex">
      {/* Sidebar Navigation */}
      <aside className="w-64 bg-slate-900/70 border-r border-slate-800 flex flex-col justify-between p-4">
        <div>
          <div className="flex items-center gap-3 px-2 py-3 mb-6">
            <div className="p-2 rounded-xl bg-indigo-600 text-white">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <div className="font-bold text-base">VisionMatch AI</div>
              <div className="text-[11px] text-slate-400">
                Multimodal VLM Engine
              </div>
            </div>
          </div>

          <div className="mb-5 px-2">
            <label className="block text-[11px] uppercase tracking-wider text-slate-400 mb-1.5">
              Active Embedding Model
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-2 text-xs text-slate-200 focus:border-indigo-500 focus:outline-none"
            >
              {models.map((m) => (
                <option key={m.model_key} value={m.model_key}>
                  {m.model_name} ({m.embedding_dimension}d)
                </option>
              ))}
            </select>
          </div>

          <nav className="space-y-1">
            {[
              { id: "dashboard", label: "System Overview", icon: Activity },
              { id: "search", label: "Semantic Search", icon: Search },
              { id: "match", label: "Cross-Modal Matcher", icon: Layers },
              { id: "upload", label: "Upload & Index", icon: Upload },
              { id: "gallery", label: "Vector Gallery", icon: ImageIcon },
              { id: "models", label: "ML Evaluation & Models", icon: BarChart3 },
              { id: "history", label: "Search Audit Log", icon: History },
            ].map((item) => {
              const Icon = item.icon;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id as any)}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-xs font-medium transition ${
                    activeTab === item.id
                      ? "bg-indigo-600 text-white"
                      : "text-slate-400 hover:bg-slate-800/60 hover:text-slate-200"
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {item.label}
                </button>
              );
            })}
          </nav>
        </div>

        <div className="border-t border-slate-800 pt-4 px-2 flex items-center justify-between">
          <div className="truncate">
            <div className="text-xs font-medium text-slate-200 truncate">
              {user?.email}
            </div>
            <div className="text-[10px] text-emerald-400">pgvector Active</div>
          </div>
          <button
            onClick={handleLogout}
            className="p-2 text-slate-400 hover:text-red-400 rounded-lg hover:bg-slate-800"
            title="Sign out"
          >
            <LogOut className="w-4 h-4" />
          </button>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 p-8 overflow-y-auto">
        {/* TAB 1: DASHBOARD */}
        {activeTab === "dashboard" && stats && (
          <div className="space-y-8">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold">
                  Multimodal Intelligence Telemetry
                </h2>
                <p className="text-sm text-slate-400">
                  Real-time pgvector index status, background Celery pipeline
                  health, and VLM latency.
                </p>
              </div>
              <button
                onClick={fetchAllData}
                className="flex items-center gap-2 px-3.5 py-2 rounded-lg bg-slate-900 border border-slate-800 text-xs hover:bg-slate-800"
              >
                <RefreshCw className="w-3.5 h-3.5" /> Refresh Telemetry
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-slate-900 border border-slate-800 p-5 rounded-xl">
                <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
                  <span>Total Indexed Images</span>
                  <Database className="w-4 h-4 text-indigo-400" />
                </div>
                <div className="text-3xl font-bold">{stats.total_images}</div>
                <div className="text-[11px] text-emerald-400 mt-1">
                  {stats.completed_jobs} completed jobs
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 p-5 rounded-xl">
                <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
                  <span>Stored Vector Embeddings</span>
                  <Cpu className="w-4 h-4 text-sky-400" />
                </div>
                <div className="text-3xl font-bold">
                  {stats.total_embeddings}
                </div>
                <div className="text-[11px] text-slate-400 mt-1">
                  L2-Normalized Hypersphere
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 p-5 rounded-xl">
                <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
                  <span>Pipeline Queue Status</span>
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                </div>
                <div className="text-3xl font-bold">{stats.pending_jobs}</div>
                <div className="text-[11px] text-slate-400 mt-1">
                  Pending / {stats.failed_jobs} Failed
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 p-5 rounded-xl">
                <div className="flex items-center justify-between text-slate-400 text-xs mb-2">
                  <span>Active VLM Encoder</span>
                  <Sparkles className="w-4 h-4 text-amber-400" />
                </div>
                <div className="text-xl font-bold font-mono truncate">
                  {selectedModel}
                </div>
                <div className="text-[11px] text-slate-400 mt-1">
                  Cosine Distance (&lt;=&gt;)
                </div>
              </div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
              <h3 className="text-sm font-semibold mb-4">
                Quick-Launch Semantic Queries
              </h3>
              <div className="flex flex-wrap gap-2">
                {[
                  "A bright red sports car parked beside a sunset ocean beach",
                  "Snow-capped mountain peaks under a clear blue alpine sky",
                  "A golden retriever dog sitting on green grass in a sunny park",
                  "Futuristic cyberpunk city street at night with glowing neon skyscrapers",
                  "Dense green pine forest trees and botanical foliage",
                ].map((sample) => (
                  <button
                    key={sample}
                    onClick={() => {
                      setQueryText(sample);
                      setSearchMode("text");
                      setActiveTab("search");
                    }}
                    className="text-xs bg-slate-950 hover:bg-indigo-600/20 border border-slate-800 hover:border-indigo-500/40 px-3 py-2 rounded-lg text-slate-300 transition"
                  >
                    "{sample}"
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: SEMANTIC SEARCH STUDIO */}
        {activeTab === "search" && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold">Multimodal Semantic Search</h2>
              <p className="text-sm text-slate-400">
                Retrieve images by natural language meaning or visual similarity
                using shared VLM vector space.
              </p>
            </div>

            <form
              onSubmit={executeSearch}
              className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4"
            >
              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={() => setSearchMode("text")}
                  className={`px-4 py-2 rounded-lg text-xs font-semibold ${
                    searchMode === "text"
                      ? "bg-indigo-600 text-white"
                      : "bg-slate-950 text-slate-400 border border-slate-800"
                  }`}
                >
                  Text-to-Image Search
                </button>
                <button
                  type="button"
                  onClick={() => setSearchMode("image")}
                  className={`px-4 py-2 rounded-lg text-xs font-semibold ${
                    searchMode === "image"
                      ? "bg-indigo-600 text-white"
                      : "bg-slate-950 text-slate-400 border border-slate-800"
                  }`}
                >
                  Image-to-Image Similarity
                </button>
              </div>

              {searchMode === "text" ? (
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={queryText}
                    onChange={(e) => setQueryText(e.target.value)}
                    placeholder="Describe a scene, object, lighting, or concept..."
                    className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-4 py-3 text-sm focus:border-indigo-500 focus:outline-none"
                  />
                  <select
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                    className="bg-slate-950 border border-slate-800 rounded-lg px-3 text-xs"
                  >
                    <option value={4}>Top 4</option>
                    <option value={8}>Top 8</option>
                    <option value={12}>Top 12</option>
                  </select>
                  <button
                    type="submit"
                    disabled={isSearching}
                    className="px-6 py-3 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-sm font-semibold text-white transition"
                  >
                    {isSearching ? "Encoding..." : "Search"}
                  </button>
                </div>
              ) : (
                <div className="flex items-center gap-4">
                  <input
                    type="file"
                    accept="image/png,image/jpeg,image/webp"
                    onChange={(e) =>
                      setSearchImageFile(e.target.files?.[0] || null)
                    }
                    className="flex-1 text-xs text-slate-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-slate-800 file:text-slate-200"
                  />
                  <button
                    type="submit"
                    disabled={!searchImageFile || isSearching}
                    className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-sm font-semibold text-white"
                  >
                    {isSearching ? "Encoding..." : "Find Similar Images"}
                  </button>
                </div>
              )}

              {searchMeta && (
                <div className="flex items-center gap-6 text-xs text-slate-400 pt-2 border-t border-slate-800/80">
                  <span>
                    Model:{" "}
                    <strong className="text-slate-200">{selectedModel}</strong>
                  </span>
                  <span>
                    Encoder Inference:{" "}
                    <strong className="text-emerald-400">
                      {searchMeta.emb_ms} ms
                    </strong>
                  </span>
                  <span>
                    pgvector ANN Retrieval:{" "}
                    <strong className="text-sky-400">
                      {searchMeta.ret_ms} ms
                    </strong>
                  </span>
                  <span>
                    Total Latency:{" "}
                    <strong className="text-slate-200">
                      {searchMeta.total_ms} ms
                    </strong>
                  </span>
                </div>
              )}
            </form>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
              {searchResults.map((item) => (
                <div
                  key={item.image_id}
                  className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden flex flex-col justify-between"
                >
                  <div>
                    <div className="relative h-48 bg-slate-950">
                      <img
                        src={
                          item.thumbnail_url.startsWith("http")
                            ? item.thumbnail_url
                            : `http://localhost:8000${item.thumbnail_url}`
                        }
                        alt={item.original_filename}
                        className="w-full h-full object-cover"
                      />
                      <span className="absolute top-2 left-2 bg-slate-950/90 border border-slate-700 text-[11px] font-mono px-2 py-0.5 rounded text-slate-200">
                        Rank #{item.rank}
                      </span>
                      <span className="absolute top-2 right-2 bg-indigo-600/95 text-[11px] font-mono px-2 py-0.5 rounded text-white">
                        cos: {item.similarity_score.toFixed(4)}
                      </span>
                    </div>
                    <div className="p-4">
                      <div className="text-xs font-semibold text-slate-200 truncate">
                        {item.original_filename}
                      </div>
                      {item.caption_text && (
                        <p className="text-[11px] text-slate-400 mt-1 line-clamp-2">
                          {item.caption_text}
                        </p>
                      )}
                    </div>
                  </div>
                  <div className="px-4 pb-3 pt-2 border-t border-slate-800/60 flex justify-between items-center">
                    <span className="text-[10px] uppercase tracking-wider text-slate-400 bg-slate-950 px-2 py-0.5 rounded">
                      {item.category || "untagged"}
                    </span>
                    <button
                      onClick={() => findSimilarByImageId(item.image_id)}
                      className="text-xs text-indigo-400 hover:text-indigo-300 font-medium"
                    >
                      Find Similar →
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* TAB 3: CROSS-MODAL IMAGE-TEXT MATCHING */}
        {activeTab === "match" && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold">
                Cross-Modal Image-Text Alignment Scorer
              </h2>
              <p className="text-sm text-slate-400">
                Select any indexed image and rank multiple candidate natural
                language hypotheses by cosine similarity.
              </p>
            </div>

            <form
              onSubmit={handleRunCrossModalMatch}
              className="grid grid-cols-1 lg:grid-cols-2 gap-6 bg-slate-900 border border-slate-800 p-6 rounded-xl"
            >
              <div className="space-y-4">
                <label className="block text-xs font-semibold text-slate-300">
                  1. Select Target Image
                </label>
                <select
                  value={matchImageId}
                  onChange={(e) => setMatchImageId(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2.5 text-xs"
                >
                  {images.map((img) => (
                    <option key={img.id} value={img.id}>
                      {img.original_filename} ({img.category || "general"})
                    </option>
                  ))}
                </select>

                {matchImageId && (
                  <div className="h-56 bg-slate-950 rounded-lg overflow-hidden border border-slate-800 flex items-center justify-center">
                    {(() => {
                      const found = images.find((i) => i.id === matchImageId);
                      if (!found) return null;
                      const src = found.image_url.startsWith("http")
                        ? found.image_url
                        : `http://localhost:8000${found.image_url}`;
                      return (
                        <img
                          src={src}
                          alt={found.original_filename}
                          className="h-full object-contain"
                        />
                      );
                    })()}
                  </div>
                )}
              </div>

              <div className="space-y-4">
                <label className="block text-xs font-semibold text-slate-300">
                  2. Candidate Text Descriptions (One per line)
                </label>
                <textarea
                  rows={6}
                  value={candidateCaptions}
                  onChange={(e) => setCandidateCaptions(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs font-mono focus:border-indigo-500 focus:outline-none"
                />
                <button
                  type="submit"
                  className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-xs font-semibold text-white"
                >
                  Compute Cosine Similarities
                </button>

                {matchRankings.length > 0 && (
                  <div className="space-y-2 pt-2">
                    {matchRankings.map((r) => (
                      <div
                        key={r.rank}
                        className="bg-slate-950 border border-slate-800 p-3 rounded-lg"
                      >
                        <div className="flex justify-between text-xs mb-1">
                          <span className="font-medium text-slate-200">
                            #{r.rank} {r.description}
                          </span>
                          <span className="font-mono text-indigo-400">
                            cos={r.cosine_similarity.toFixed(4)} (
                            {r.relative_alignment_pct}%)
                          </span>
                        </div>
                        <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                          <div
                            className="bg-indigo-500 h-full"
                            style={{ width: `${r.relative_alignment_pct}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </form>
          </div>
        )}

        {/* TAB 4: UPLOAD IMAGE */}
        {activeTab === "upload" && (
          <div className="max-w-2xl space-y-6">
            <div>
              <h2 className="text-2xl font-bold">
                Upload & Asynchronous VLM Indexing
              </h2>
              <p className="text-sm text-slate-400">
                Validates magic bytes, strips EXIF metadata, stores in MinIO S3,
                and dispatches an idempotent Celery embedding job.
              </p>
            </div>

            <form
              onSubmit={handleUploadImage}
              className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4"
            >
              <div>
                <label className="block text-xs text-slate-400 mb-2">
                  Select Image File (JPEG, PNG, WEBP up to 10MB)
                </label>
                <input
                  type="file"
                  required
                  accept="image/jpeg,image/png,image/webp"
                  onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                  className="w-full text-xs text-slate-300 file:mr-4 file:py-2.5 file:px-4 file:rounded-lg file:border-0 file:bg-indigo-600 file:text-white"
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs text-slate-400 mb-1">
                    Semantic Category
                  </label>
                  <input
                    type="text"
                    value={uploadCategory}
                    onChange={(e) => setUploadCategory(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs"
                  />
                </div>
                <div>
                  <label className="block text-xs text-slate-400 mb-1">
                    Optional Ground-Truth Caption
                  </label>
                  <input
                    type="text"
                    value={uploadCaption}
                    onChange={(e) => setUploadCaption(e.target.value)}
                    placeholder="Used for offline Recall@K evaluation"
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs"
                  />
                </div>
              </div>
              <button
                type="submit"
                className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-xs font-semibold text-white"
              >
                Upload & Generate Vector Embedding
              </button>

              {uploadStatus && (
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs text-emerald-400">
                  {uploadStatus}
                </div>
              )}
            </form>
          </div>
        )}

        {/* TAB 5: GALLERY */}
        {activeTab === "gallery" && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold">
                Indexed Multimodal Asset Library ({images.length})
              </h2>
              <p className="text-sm text-slate-400">
                Inspect image dimensions, processing status, and trigger visual
                similarity retrieval.
              </p>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
              {images.map((img) => (
                <div
                  key={img.id}
                  className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden flex flex-col justify-between"
                >
                  <div>
                    <div className="relative h-44 bg-slate-950">
                      <img
                        src={
                          img.thumbnail_url.startsWith("http")
                            ? img.thumbnail_url
                            : `http://localhost:8000${img.thumbnail_url}`
                        }
                        alt={img.original_filename}
                        className="w-full h-full object-cover"
                      />
                      <span className="absolute top-2 right-2 bg-emerald-600/90 text-[10px] px-2 py-0.5 rounded text-white font-semibold">
                        {img.processing_status}
                      </span>
                    </div>
                    <div className="p-4">
                      <div className="text-xs font-semibold truncate">
                        {img.original_filename}
                      </div>
                      <div className="text-[11px] text-slate-400 mt-1">
                        {img.width}x{img.height}px •{" "}
                        {(img.file_size_bytes / 1024).toFixed(1)} KB
                      </div>
                      {img.caption_text && (
                        <p className="text-[11px] text-slate-400 mt-1 line-clamp-2">
                          {img.caption_text}
                        </p>
                      )}
                    </div>
                  </div>
                  <div className="px-4 py-2.5 border-t border-slate-800 flex justify-between items-center">
                    <button
                      onClick={() => findSimilarByImageId(img.id)}
                      className="text-xs text-indigo-400 hover:text-indigo-300"
                    >
                      Search Similar
                    </button>
                    <button
                      onClick={() => handleDeleteImage(img.id)}
                      className="text-slate-500 hover:text-red-400"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* TAB 6: MODEL COMPARISON & ML EVALUATION */}
        {activeTab === "models" && (
          <div className="space-y-8">
            <div>
              <h2 className="text-2xl font-bold">
                Model Comparison & Scientific IR Evaluation
              </h2>
              <p className="text-sm text-slate-400">
                Benchmark Vision-Language Models on identical ground-truth
                queries for Recall@K, Precision@5, mAP, MRR, and inference
                latency.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
              {models.map((m) => (
                <div
                  key={m.model_key}
                  className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col justify-between space-y-4"
                >
                  <div>
                    <div className="flex justify-between items-start">
                      <h3 className="font-bold text-base">{m.model_name}</h3>
                      <span className="text-xs font-mono bg-indigo-500/10 text-indigo-400 border border-indigo-500/30 px-2 py-0.5 rounded">
                        {m.embedding_dimension}d
                      </span>
                    </div>
                    <div className="text-xs font-mono text-slate-400 mt-1">
                      {m.hf_model_id}
                    </div>
                    <div className="mt-4 text-xs text-slate-300">
                      Indexed Vectors:{" "}
                      <strong>{m.indexed_image_count}</strong>
                    </div>
                  </div>
                  <button
                    disabled={isEvalRunning}
                    onClick={() => handleRunEvaluation(m.model_key)}
                    className="w-full flex items-center justify-center gap-2 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 rounded-lg text-xs font-semibold text-white transition"
                  >
                    <Play className="w-3.5 h-3.5" />
                    {isEvalRunning
                      ? "Running IR Benchmark..."
                      : `Evaluate ${m.model_name}`}
                  </button>
                </div>
              ))}
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
              <div className="p-4 border-b border-slate-800 font-semibold text-sm">
                Historical Information Retrieval Benchmark Runs (Logged to
                PostgreSQL & MLflow)
              </div>
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 bg-slate-950/60">
                    <th className="p-3.5">Model</th>
                    <th className="p-3.5">Dataset</th>
                    <th className="p-3.5">Recall@1</th>
                    <th className="p-3.5">Recall@5</th>
                    <th className="p-3.5">Recall@10</th>
                    <th className="p-3.5">Precision@5</th>
                    <th className="p-3.5">mAP</th>
                    <th className="p-3.5">MRR</th>
                    <th className="p-3.5">Avg Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {evaluations.map((ev) => (
                    <tr
                      key={ev.id}
                      className="border-b border-slate-800/60 hover:bg-slate-800/30"
                    >
                      <td className="p-3.5 font-semibold text-slate-200">
                        {ev.model_name}
                      </td>
                      <td className="p-3.5 font-mono text-slate-400">
                        {ev.dataset_name} ({ev.num_queries}q)
                      </td>
                      <td className="p-3.5 font-mono text-emerald-400">
                        {(ev.recall_at_1 * 100).toFixed(1)}%
                      </td>
                      <td className="p-3.5 font-mono text-emerald-400">
                        {(ev.recall_at_5 * 100).toFixed(1)}%
                      </td>
                      <td className="p-3.5 font-mono">
                        {(ev.recall_at_10 * 100).toFixed(1)}%
                      </td>
                      <td className="p-3.5 font-mono">
                        {(ev.precision_at_5 * 100).toFixed(1)}%
                      </td>
                      <td className="p-3.5 font-mono font-bold text-indigo-400">
                        {ev.mean_average_precision.toFixed(4)}
                      </td>
                      <td className="p-3.5 font-mono font-bold text-sky-400">
                        {ev.mean_reciprocal_rank.toFixed(4)}
                      </td>
                      <td className="p-3.5 font-mono text-amber-400">
                        {ev.average_latency_ms} ms
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* TAB 7: SEARCH HISTORY */}
        {activeTab === "history" && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
            <table className="w-full text-left border-collapse text-sm">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 bg-slate-950/60">
                  <th className="p-4">Query Type</th>
                  <th className="p-4">Query Text</th>
                  <th className="p-4">Model</th>
                  <th className="p-4">Results</th>
                  <th className="p-4">Latency</th>
                  <th className="p-4">Action</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h) => (
                  <tr
                    key={h.id}
                    className="border-b border-slate-800/60 hover:bg-slate-800/30"
                  >
                    <td className="p-4">
                      <span className="text-xs px-2.5 py-1 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                        {h.query_type}
                      </span>
                    </td>
                    <td className="p-4 font-medium text-slate-200">
                      {h.query_text || "Image Query"}
                    </td>
                    <td className="p-4 font-mono text-xs text-slate-400">
                      {h.model_key}
                    </td>
                    <td className="p-4">{h.result_count}</td>
                    <td className="p-4 text-emerald-400 font-mono">
                      {h.latency_ms} ms
                    </td>
                    <td className="p-4">
                      {h.query_type === "TEXT_TO_IMAGE" && h.query_text && (
                        <button
                          onClick={() => {
                            setQueryText(h.query_text || "");
                            setSelectedModel(h.model_key);
                            setSearchMode("text");
                            setActiveTab("search");
                          }}
                          className="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-indigo-600 text-slate-200 transition"
                        >
                          Replay Search
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
                {history.length === 0 && (
                  <tr>
                    <td colSpan={6} className="p-8 text-center text-slate-500">
                      No searches recorded yet. Run a text or image search to
                      populate history.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </main>
    </div>
  );
}