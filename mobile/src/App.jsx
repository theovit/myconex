import React, {
  useCallback, useEffect, useMemo, useRef, useState,
} from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  Activity, Bot, ChevronUp, Cpu, Globe, Lock, Mic, MicOff,
  Network, Radio, Send, Shield, Wifi, WifiOff, Zap,
} from 'lucide-react'

// ─── Constants ────────────────────────────────────────────────────────────────

const MESH_API = import.meta.env.VITE_MESH_API || 'http://localhost:8765'
const WS_URL   = import.meta.env.VITE_WS_URL   || 'ws://localhost:8765/ws'

const TIER_COLORS = {
  T1: { fg: '#A78BFA', glow: '#A78BFA50', label: 'Apex' },
  T2: { fg: '#00FFB2', glow: '#00FFB250', label: 'Mid-GPU' },
  T3: { fg: '#0EA5E9', glow: '#0EA5E950', label: 'CPU-Heavy' },
  T4: { fg: '#F59E0B', glow: '#F59E0B50', label: 'Edge' },
}

// ─── Mycelium Canvas ─────────────────────────────────────────────────────────

function MyceliumCanvas({ nodes }) {
  const canvasRef = useRef(null)
  const animRef   = useRef(null)
  const stateRef  = useRef({ spores: [], pulses: [], t: 0 })

  // Build spore positions from real nodes or generate placeholders
  const buildSpores = useCallback((w, h, nodeList) => {
    const count = Math.max(nodeList.length, 8)
    return Array.from({ length: count }, (_, i) => {
      const node = nodeList[i]
      const angle = (i / count) * Math.PI * 2 + Math.random() * 0.4
      const r = 0.28 + Math.random() * 0.18
      const tc = TIER_COLORS[node?.tier || 'T3']
      return {
        x: w * 0.5 + Math.cos(angle) * w * r,
        y: h * 0.5 + Math.sin(angle) * h * r,
        r: node ? 5 + (5 - parseInt(node.tier?.[1] || 3)) * 2 : 3,
        color: tc.fg,
        glow: tc.glow,
        name: node?.name || '',
        online: node?.is_online ?? true,
        phase: Math.random() * Math.PI * 2,
        vx: (Math.random() - 0.5) * 0.12,
        vy: (Math.random() - 0.5) * 0.12,
      }
    })
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width = canvas.offsetWidth * devicePixelRatio
    const H = canvas.height = canvas.offsetHeight * devicePixelRatio
    ctx.scale(devicePixelRatio, devicePixelRatio)
    const w = W / devicePixelRatio
    const h = H / devicePixelRatio

    stateRef.current.spores = buildSpores(w, h, nodes)

    const draw = () => {
      const { spores, pulses, t: time } = stateRef.current
      stateRef.current.t += 0.008

      ctx.clearRect(0, 0, w, h)

      // Background radial gradient — bioluminescent abyss
      const grad = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w * 0.6)
      grad.addColorStop(0,   'rgba(0,255,178,0.03)')
      grad.addColorStop(0.5, 'rgba(14,165,233,0.015)')
      grad.addColorStop(1,   'rgba(0,0,0,0)')
      ctx.fillStyle = grad
      ctx.fillRect(0, 0, w, h)

      // Draw hyphae (edges)
      for (let i = 0; i < spores.length; i++) {
        for (let j = i + 1; j < spores.length; j++) {
          const a = spores[i], b = spores[j]
          const dx = b.x - a.x, dy = b.y - a.y
          const dist = Math.sqrt(dx * dx + dy * dy)
          const maxDist = w * 0.38
          if (dist > maxDist) continue

          const alpha = (1 - dist / maxDist) * 0.35
          const pulse = 0.5 + 0.5 * Math.sin(time * 1.2 + a.phase + b.phase)

          ctx.beginPath()
          ctx.moveTo(a.x, a.y)

          // Slightly curved hyphae
          const mx = (a.x + b.x) / 2 + Math.sin(time + i) * 8
          const my = (a.y + b.y) / 2 + Math.cos(time + j) * 8
          ctx.quadraticCurveTo(mx, my, b.x, b.y)

          const lineGrad = ctx.createLinearGradient(a.x, a.y, b.x, b.y)
          lineGrad.addColorStop(0, a.color + Math.floor(alpha * 255 * pulse).toString(16).padStart(2, '0'))
          lineGrad.addColorStop(1, b.color + Math.floor(alpha * 255 * pulse).toString(16).padStart(2, '0'))
          ctx.strokeStyle = lineGrad
          ctx.lineWidth = 0.6 + pulse * 0.6
          ctx.stroke()
        }
      }

      // Draw signal pulses traveling along edges
      stateRef.current.pulses = pulses.filter(p => p.progress < 1)
      for (const p of stateRef.current.pulses) {
        p.progress += 0.018
        const x = p.x0 + (p.x1 - p.x0) * p.progress
        const y = p.y0 + (p.y1 - p.y0) * p.progress
        ctx.beginPath()
        ctx.arc(x, y, 3, 0, Math.PI * 2)
        ctx.fillStyle = p.color
        ctx.shadowBlur = 10
        ctx.shadowColor = p.color
        ctx.fill()
        ctx.shadowBlur = 0
      }

      // Spawn new pulses occasionally
      if (Math.random() < 0.06 && spores.length > 1) {
        const a = spores[Math.floor(Math.random() * spores.length)]
        const b = spores[Math.floor(Math.random() * spores.length)]
        if (a !== b) {
          stateRef.current.pulses.push({
            x0: a.x, y0: a.y, x1: b.x, y1: b.y,
            progress: 0, color: a.color,
          })
        }
      }

      // Draw spore nodes
      for (const s of spores) {
        // Drift
        s.x += s.vx
        s.y += s.vy
        if (s.x < 20 || s.x > w - 20) s.vx *= -1
        if (s.y < 20 || s.y > h - 20) s.vy *= -1

        const pulse = 0.7 + 0.3 * Math.sin(time * 2 + s.phase)
        const glowSize = s.r * 4 * pulse

        // Outer glow
        const glow = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, glowSize)
        glow.addColorStop(0, s.color + 'AA')
        glow.addColorStop(1, s.color + '00')
        ctx.beginPath()
        ctx.arc(s.x, s.y, glowSize, 0, Math.PI * 2)
        ctx.fillStyle = glow
        ctx.fill()

        // Core
        ctx.beginPath()
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2)
        ctx.fillStyle = s.online ? s.color : '#334155'
        ctx.shadowBlur = s.online ? 14 : 0
        ctx.shadowColor = s.color
        ctx.fill()
        ctx.shadowBlur = 0
      }

      animRef.current = requestAnimationFrame(draw)
    }

    draw()
    return () => cancelAnimationFrame(animRef.current)
  }, [nodes, buildSpores])

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full"
      style={{ opacity: 0.9 }}
    />
  )
}

// ─── Tier Badge ───────────────────────────────────────────────────────────────

function TierBadge({ tier }) {
  const tc = TIER_COLORS[tier] || TIER_COLORS.T3
  return (
    <span
      className="text-[10px] font-bold px-1.5 py-0.5 rounded"
      style={{ color: tc.fg, background: tc.glow, border: `1px solid ${tc.fg}40` }}
    >
      {tier}
    </span>
  )
}

// ─── Node Card ────────────────────────────────────────────────────────────────

function NodeCard({ node }) {
  const tc = TIER_COLORS[node.tier] || TIER_COLORS.T3
  return (
    <motion.div
      initial={{ opacity: 0, x: -12 }}
      animate={{ opacity: 1, x: 0 }}
      className="flex items-center gap-3 px-3 py-2.5 rounded-xl glass"
      style={{ border: `1px solid ${tc.fg}25` }}
    >
      <div
        className="w-2.5 h-2.5 rounded-full flex-shrink-0"
        style={{
          background: node.is_online ? tc.fg : '#475569',
          boxShadow: node.is_online ? `0 0 8px ${tc.fg}` : 'none',
        }}
      />
      <div className="min-w-0 flex-1">
        <div className="text-xs font-semibold text-slate-200 truncate">{node.name}</div>
        <div className="text-[10px] text-slate-500 truncate">
          {node.address} · {(node.roles || []).join(', ')}
        </div>
      </div>
      <TierBadge tier={node.tier} />
    </motion.div>
  )
}

// ─── Tunnel Status ────────────────────────────────────────────────────────────

function TunnelStatus({ connected, nodeCount }) {
  return (
    <div className="flex items-center gap-2">
      <div className="relative flex items-center gap-1.5">
        {connected ? (
          <>
            <Lock size={12} className="text-emerald-400" />
            <span className="text-[11px] text-emerald-400 font-medium glow-green">
              E2E ENCRYPTED
            </span>
            <span className="text-[11px] text-slate-500">·</span>
            <span className="text-[11px] text-slate-400">{nodeCount} nodes</span>
          </>
        ) : (
          <>
            <WifiOff size={12} className="text-slate-500" />
            <span className="text-[11px] text-slate-500">MESH OFFLINE</span>
          </>
        )}
      </div>
    </div>
  )
}

// ─── Chat Bubble ──────────────────────────────────────────────────────────────

function ChatBubble({ msg }) {
  const isUser = msg.role === 'user'
  return (
    <div className={`flex msg-enter ${isUser ? 'justify-end' : 'justify-start'} mb-3`}>
      {!isUser && (
        <div
          className="w-7 h-7 rounded-full flex items-center justify-center mr-2 flex-shrink-0 mt-0.5"
          style={{ background: '#00FFB215', border: '1px solid #00FFB240' }}
        >
          <Bot size={14} style={{ color: '#00FFB2' }} />
        </div>
      )}
      <div
        className={`max-w-[78%] px-4 py-2.5 rounded-2xl text-sm leading-relaxed ${
          isUser
            ? 'bg-sky-600/30 text-sky-100 rounded-tr-sm'
            : 'glass text-slate-200 rounded-tl-sm'
        }`}
        style={!isUser ? { border: '1px solid #00FFB220' } : {}}
      >
        {msg.content}
        {msg.model && (
          <div className="text-[10px] text-slate-500 mt-1">{msg.model} · {msg.node}</div>
        )}
      </div>
    </div>
  )
}

// ─── Voice Button ─────────────────────────────────────────────────────────────

function VoiceButton({ listening, onToggle }) {
  return (
    <motion.button
      whileTap={{ scale: 0.9 }}
      onClick={onToggle}
      className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0"
      style={{
        background: listening ? '#00FFB220' : '#0D1B26',
        border: `1px solid ${listening ? '#00FFB280' : '#1A3045'}`,
        boxShadow: listening ? '0 0 16px #00FFB240' : 'none',
      }}
    >
      <AnimatePresence mode="wait">
        {listening ? (
          <motion.div
            key="on"
            initial={{ scale: 0.7, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.7, opacity: 0 }}
          >
            <MicOff size={16} style={{ color: '#00FFB2' }} />
          </motion.div>
        ) : (
          <motion.div
            key="off"
            initial={{ scale: 0.7, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.7, opacity: 0 }}
          >
            <Mic size={16} className="text-slate-400" />
          </motion.div>
        )}
      </AnimatePresence>
    </motion.button>
  )
}

// ─── Tabs ─────────────────────────────────────────────────────────────────────

const TABS = [
  { id: 'chat',    icon: Bot,      label: 'Chat' },
  { id: 'mesh',    icon: Network,  label: 'Mesh' },
  { id: 'status',  icon: Activity, label: 'Status' },
]

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [tab, setTab]               = useState('chat')
  const [messages, setMessages]     = useState([
    {
      role: 'assistant',
      content: 'MYCONEX online. I am connected to the mesh. How can I assist?',
      model: 'phi3:mini',
      node: 'local',
    },
  ])
  const [input, setInput]           = useState('')
  const [listening, setListening]   = useState(false)
  const [loading, setLoading]       = useState(false)
  const [peers, setPeers]           = useState([])
  const [localStatus, setLocalStatus] = useState(null)
  const [meshOnline, setMeshOnline] = useState(false)
  const [panelOpen, setPanelOpen]   = useState(false)
  const messagesEndRef              = useRef(null)
  const inputRef                    = useRef(null)
  const recognitionRef              = useRef(null)

  // ─── Fetch mesh data ────────────────────────────────────────────────────────

  const fetchStatus = useCallback(async () => {
    try {
      const [statusRes, peersRes] = await Promise.all([
        fetch(`${MESH_API}/status`),
        fetch(`${MESH_API}/peers`),
      ])
      if (statusRes.ok) {
        setLocalStatus(await statusRes.json())
        setMeshOnline(true)
      }
      if (peersRes.ok) setPeers(await peersRes.json())
    } catch {
      setMeshOnline(false)
    }
  }, [])

  useEffect(() => {
    fetchStatus()
    const id = setInterval(fetchStatus, 8000)
    return () => clearInterval(id)
  }, [fetchStatus])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // ─── Send message ────────────────────────────────────────────────────────────

  const sendMessage = useCallback(async (text) => {
    const trimmed = text.trim()
    if (!trimmed || loading) return

    setMessages(prev => [...prev, { role: 'user', content: trimmed }])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch(`${MESH_API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: trimmed }),
      })

      if (res.ok) {
        const data = await res.json()
        const reply = data.response?.response || data.response || 'No response.'
        setMessages(prev => [
          ...prev,
          {
            role: 'assistant',
            content: reply,
            model: data.model || 'unknown',
            node: data.agent || 'mesh',
          },
        ])
      } else {
        throw new Error(`HTTP ${res.status}`)
      }
    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: `⚠ Mesh unreachable: ${err.message}. Running in offline mode.`,
          model: '',
          node: '',
        },
      ])
    } finally {
      setLoading(false)
    }
  }, [loading])

  // ─── Voice input ─────────────────────────────────────────────────────────────

  const toggleVoice = useCallback(() => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      alert('Speech recognition not supported on this device.')
      return
    }

    if (listening) {
      recognitionRef.current?.stop()
      setListening(false)
      return
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    const rec = new SpeechRecognition()
    rec.continuous = false
    rec.interimResults = true
    rec.lang = 'en-US'

    rec.onresult = (e) => {
      const transcript = Array.from(e.results)
        .map(r => r[0].transcript)
        .join('')
      setInput(transcript)
      if (e.results[e.results.length - 1].isFinal) {
        sendMessage(transcript)
        setListening(false)
      }
    }
    rec.onerror = () => setListening(false)
    rec.onend = () => setListening(false)

    recognitionRef.current = rec
    rec.start()
    setListening(true)
  }, [listening, sendMessage])

  // ─── All nodes including self ─────────────────────────────────────────────

  const allNodes = useMemo(() => {
    const self = localStatus
      ? [{
          name: localStatus.node || 'this-node',
          tier: localStatus.tier || 'T3',
          roles: localStatus.roles || [],
          address: 'localhost',
          is_online: true,
        }]
      : []
    return [...self, ...peers]
  }, [localStatus, peers])

  // ─── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="relative flex flex-col h-full w-full overflow-hidden" style={{ background: '#050A0E' }}>

      {/* ── Mycelium background canvas ─────────────────────────────────── */}
      <div className="absolute inset-0 pointer-events-none">
        <MyceliumCanvas nodes={allNodes} />
      </div>

      {/* ── Header ────────────────────────────────────────────────────── */}
      <header className="relative z-10 flex items-center justify-between px-4 pt-safe pt-3 pb-3 glass border-b border-white/5">
        <div className="flex items-center gap-2.5">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ background: '#00FFB215', border: '1px solid #00FFB250' }}
          >
            <Radio size={16} style={{ color: '#00FFB2' }} />
          </div>
          <div>
            <div className="text-sm font-bold text-white tracking-wide glow-green">MYCONEX</div>
            <TunnelStatus connected={meshOnline} nodeCount={allNodes.length} />
          </div>
        </div>

        <div className="flex items-center gap-2">
          {localStatus && <TierBadge tier={localStatus.tier} />}
          <motion.button
            whileTap={{ scale: 0.9 }}
            onClick={() => { setPanelOpen(p => !p); setTab('mesh') }}
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ background: '#0D1B26', border: '1px solid #1A3045' }}
          >
            <Globe size={15} className="text-slate-400" />
          </motion.button>
        </div>
      </header>

      {/* ── Tab content ───────────────────────────────────────────────── */}
      <main className="relative z-10 flex-1 overflow-hidden flex flex-col">
        <AnimatePresence mode="wait">

          {/* CHAT TAB */}
          {tab === 'chat' && (
            <motion.div
              key="chat"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex-1 flex flex-col overflow-hidden"
            >
              {/* Messages */}
              <div className="flex-1 overflow-y-auto px-4 py-4">
                {messages.map((msg, i) => (
                  <ChatBubble key={i} msg={msg} />
                ))}

                {/* Loading indicator */}
                {loading && (
                  <div className="flex items-center gap-2 mb-3 msg-enter">
                    <div
                      className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0"
                      style={{ background: '#00FFB215', border: '1px solid #00FFB240' }}
                    >
                      <Bot size={14} style={{ color: '#00FFB2' }} />
                    </div>
                    <div className="glass px-4 py-3 rounded-2xl rounded-tl-sm flex gap-1.5"
                         style={{ border: '1px solid #00FFB220' }}>
                      {[0, 1, 2].map(i => (
                        <motion.div
                          key={i}
                          className="w-1.5 h-1.5 rounded-full"
                          style={{ background: '#00FFB2' }}
                          animate={{ opacity: [0.3, 1, 0.3], y: [0, -4, 0] }}
                          transition={{ duration: 1, repeat: Infinity, delay: i * 0.15 }}
                        />
                      ))}
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input bar */}
              <div className="px-4 pb-safe pb-3 pt-2 glass border-t border-white/5">
                <div
                  className="flex items-center gap-2 px-3 py-2 rounded-2xl"
                  style={{ background: '#0D1B26', border: '1px solid #1A3045' }}
                >
                  <VoiceButton listening={listening} onToggle={toggleVoice} />
                  <input
                    ref={inputRef}
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendMessage(input)}
                    placeholder="Ask the mesh anything..."
                    className="flex-1 bg-transparent text-sm text-slate-200 placeholder-slate-600 outline-none"
                    style={{ userSelect: 'text' }}
                  />
                  <motion.button
                    whileTap={{ scale: 0.88 }}
                    onClick={() => sendMessage(input)}
                    disabled={!input.trim() || loading}
                    className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 disabled:opacity-30"
                    style={{
                      background: input.trim() ? '#00FFB220' : '#1A3045',
                      border: `1px solid ${input.trim() ? '#00FFB260' : '#1A3045'}`,
                    }}
                  >
                    <Send size={15} style={{ color: input.trim() ? '#00FFB2' : '#475569' }} />
                  </motion.button>
                </div>

                {listening && (
                  <motion.div
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-center gap-2 mt-2 px-2"
                  >
                    <motion.div
                      className="w-2 h-2 rounded-full"
                      style={{ background: '#00FFB2' }}
                      animate={{ scale: [1, 1.5, 1] }}
                      transition={{ duration: 0.8, repeat: Infinity }}
                    />
                    <span className="text-[11px] text-emerald-400">Listening...</span>
                  </motion.div>
                )}
              </div>
            </motion.div>
          )}

          {/* MESH TAB */}
          {tab === 'mesh' && (
            <motion.div
              key="mesh"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="flex-1 overflow-y-auto px-4 py-4"
            >
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                  <Network size={14} style={{ color: '#00FFB2' }} />
                  Mesh Topology
                </h2>
                <span className="text-xs text-slate-500">{allNodes.length} nodes</span>
              </div>

              <div className="space-y-2">
                {allNodes.length === 0 ? (
                  <div className="text-center py-12 text-slate-600 text-sm">
                    No peers discovered yet
                  </div>
                ) : (
                  allNodes.map((node, i) => <NodeCard key={node.name || i} node={node} />)
                )}
              </div>

              {localStatus && (
                <div className="mt-6 space-y-2">
                  <h3 className="text-xs text-slate-500 uppercase tracking-widest mb-3">Local Node</h3>
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      { label: 'Tasks Active', value: localStatus.tasks_active ?? 0, icon: Zap },
                      { label: 'Peers Online', value: localStatus.nodes_online ?? 0, icon: Wifi },
                      { label: 'NATS', value: localStatus.nats_connected ? 'Connected' : 'Offline', icon: Radio },
                      { label: 'Tier', value: localStatus.tier, icon: Cpu },
                    ].map(({ label, value, icon: Icon }) => (
                      <div key={label} className="glass rounded-xl p-3" style={{ border: '1px solid #1A3045' }}>
                        <div className="flex items-center gap-1.5 mb-1">
                          <Icon size={11} className="text-slate-500" />
                          <span className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</span>
                        </div>
                        <div className="text-sm font-semibold text-slate-200">{String(value)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {/* STATUS TAB */}
          {tab === 'status' && (
            <motion.div
              key="status"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="flex-1 overflow-y-auto px-4 py-4"
            >
              <h2 className="text-sm font-semibold text-slate-300 flex items-center gap-2 mb-4">
                <Activity size={14} style={{ color: '#00FFB2' }} />
                Node Status
              </h2>

              {/* Encryption badge */}
              <div
                className="flex items-center gap-3 p-4 rounded-xl mb-4 glass"
                style={{ border: '1px solid #00FFB230' }}
              >
                <Shield size={20} style={{ color: '#00FFB2' }} />
                <div>
                  <div className="text-sm font-semibold text-emerald-300">End-to-End Encrypted</div>
                  <div className="text-[11px] text-slate-500 mt-0.5">
                    All mesh traffic encrypted in transit · mTLS + NATS TLS
                  </div>
                </div>
              </div>

              {/* Tier breakdown */}
              <div className="mb-4">
                <h3 className="text-xs text-slate-500 uppercase tracking-widest mb-3">Tier Distribution</h3>
                <div className="space-y-2">
                  {Object.entries(TIER_COLORS).map(([tier, tc]) => {
                    const count = allNodes.filter(n => n.tier === tier).length
                    return (
                      <div key={tier} className="flex items-center gap-3">
                        <span className="text-[11px] w-6" style={{ color: tc.fg }}>{tier}</span>
                        <div className="flex-1 h-1.5 rounded-full" style={{ background: '#1A3045' }}>
                          <motion.div
                            className="h-full rounded-full"
                            style={{ background: tc.fg, boxShadow: `0 0 8px ${tc.fg}` }}
                            initial={{ width: 0 }}
                            animate={{ width: `${Math.min(100, (count / Math.max(allNodes.length, 1)) * 100)}%` }}
                            transition={{ duration: 0.8, ease: 'easeOut' }}
                          />
                        </div>
                        <span className="text-[11px] text-slate-500 w-4 text-right">{count}</span>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Connection info */}
              <div className="glass rounded-xl p-4" style={{ border: '1px solid #1A3045' }}>
                <div className="text-xs text-slate-500 uppercase tracking-widest mb-3">Connection</div>
                <div className="space-y-2 text-sm">
                  {[
                    ['API',       MESH_API,      meshOnline ? '#00FFB2' : '#EF4444'],
                    ['mDNS',      '_ai-mesh._tcp', '#0EA5E9'],
                    ['Protocol', 'NATS + mTLS',   '#A78BFA'],
                    ['Discovery', 'Zeroconf',      '#F59E0B'],
                  ].map(([label, value, color]) => (
                    <div key={label} className="flex justify-between items-center">
                      <span className="text-slate-500">{label}</span>
                      <span className="text-xs font-mono" style={{ color }}>{value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

        </AnimatePresence>
      </main>

      {/* ── Bottom nav ────────────────────────────────────────────────── */}
      <nav className="relative z-10 flex items-center justify-around px-2 pb-safe pb-3 pt-2 glass border-t border-white/5">
        {TABS.map(({ id, icon: Icon, label }) => {
          const active = tab === id
          return (
            <motion.button
              key={id}
              whileTap={{ scale: 0.88 }}
              onClick={() => setTab(id)}
              className="flex flex-col items-center gap-1 px-6 py-1.5 rounded-xl transition-colors"
              style={{
                background: active ? '#00FFB210' : 'transparent',
                border: active ? '1px solid #00FFB230' : '1px solid transparent',
              }}
            >
              <Icon
                size={18}
                style={{ color: active ? '#00FFB2' : '#475569' }}
              />
              <span
                className="text-[10px] font-medium"
                style={{ color: active ? '#00FFB2' : '#475569' }}
              >
                {label}
              </span>
            </motion.button>
          )
        })}
      </nav>

    </div>
  )
}
