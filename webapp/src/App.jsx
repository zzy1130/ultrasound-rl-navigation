import { useState, useRef, useEffect, useCallback } from 'react'
import * as d3 from 'd3'
import { 
  Upload, Play, Network, Target, Activity, Zap, CheckCircle, RotateCcw, 
  Pause, ChevronLeft, ChevronRight, Shield, AlertTriangle, FileCheck,
  Grid3X3, Stethoscope, Brain, TrendingUp
} from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8765'

function App() {
  const [imageFile, setImageFile] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [rolloutNum, setRolloutNum] = useState(100)
  const [maxSteps, setMaxSteps] = useState(50)
  const [stochasticPolicy, setStochasticPolicy] = useState(true)
  const [temperature, setTemperature] = useState(1.0)
  const [slipProb, setSlipProb] = useState(0.0)  // environment stochasticity
  const [stateMode, setStateMode] = useState('distance')  // 'grid' or 'distance' (view-based)
  const [targetRadius, setTargetRadius] = useState(10)
  const [successOnly, setSuccessOnly] = useState(false)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  
  // Tab state
  const [activeTab, setActiveTab] = useState('navigation')
  
  // Playback state
  const [selectedRollout, setSelectedRollout] = useState(0)
  const [currentStep, setCurrentStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [playbackSpeed, setPlaybackSpeed] = useState(200)
  const [showRecoveryEdges, setShowRecoveryEdges] = useState(false)
  const [showPathInNetwork, setShowPathInNetwork] = useState(false)  // Toggle for showing path in network
  const [navViewMode, setNavViewMode] = useState('playback')  // 'playback' or 'network' (state network overlay)
  
  const canvasRef = useRef(null)
  const networkRef = useRef(null)
  const animationRef = useRef(null)

  // Handle image upload
  const handleImageChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setImageFile(file)
      const reader = new FileReader()
      reader.onload = (e) => setImagePreview(e.target.result)
      reader.readAsDataURL(file)
      setResults(null)
      setError(null)
    }
  }

  // Run simulation
  const runSimulation = async () => {
    if (!imageFile) {
      setError('Please upload an ultrasound image first')
      return
    }
    
    setLoading(true)
    setError(null)
    setResults(null)
    setIsPlaying(false)
    setSelectedRollout(0)
    setCurrentStep(0)
    
    try {
      const formData = new FormData()
      formData.append('image', imageFile)
      formData.append('rollout_num', rolloutNum)
      formData.append('max_steps', maxSteps)
      formData.append('stochastic_policy', stochasticPolicy.toString())
      formData.append('temperature', temperature.toString())
      formData.append('slip_prob', slipProb.toString())
      formData.append('state_mode', stateMode)
      formData.append('target_radius', targetRadius.toString())
      formData.append('success_only', successOnly.toString())
      
      const response = await fetch(`${API_BASE}/simulate`, {
        method: 'POST',
        body: formData
      })
      
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }
      
      const data = await response.json()
      setResults(data)
      
      if (data.all_trajectories?.length > 0) {
        setSelectedRollout(0)
        setCurrentStep(0)
        setIsPlaying(true)
      }
    } catch (err) {
      setError('Simulation failed: ' + err.message)
    }
    setLoading(false)
  }

  // Get current trajectory
  const currentTrajectory = results?.all_trajectories?.[selectedRollout]?.steps || []

  // Compute network statistics (nodes with edges only)
  const networkStats = (() => {
    if (!results?.analysis?.transition_analysis) return { nodeCount: 0, edgeCount: 0, criticalCount: 0 }
    const graph = results.analysis.transition_analysis
    const nodesWithEdges = new Set()
    let edgeCount = 0
    let criticalCount = 0
    
    graph.forEach(t => {
      if (t.from_state !== t.to_state) {
        nodesWithEdges.add(t.from_state)
        nodesWithEdges.add(t.to_state)
        edgeCount++
        if (t.is_critical) criticalCount++
      }
    })
    
    return { nodeCount: nodesWithEdges.size, edgeCount, criticalCount }
  })()

  // Animation loop
  useEffect(() => {
    if (isPlaying && currentTrajectory.length > 0) {
      animationRef.current = setInterval(() => {
        setCurrentStep(prev => {
          if (prev >= currentTrajectory.length - 1) {
            setIsPlaying(false)
            return prev
          }
          return prev + 1
        })
      }, playbackSpeed)
    }
    return () => clearInterval(animationRef.current)
  }, [isPlaying, currentTrajectory.length, playbackSpeed])

  // Draw navigation on canvas
  const drawNavigation = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !imagePreview) return
    
    const ctx = canvas.getContext('2d')
    const img = new Image()
    img.onload = () => {
      canvas.width = 420
      canvas.height = 420
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
      
      if (!results) return
      
      const imgW = results.image_size?.[0] || 256
      const imgH = results.image_size?.[1] || 256
      const scaleX = canvas.width / imgW
      const scaleY = canvas.height / imgH
      const viewSize = results.view_size || 64
      
      // Draw grid lines based on actual grid dimensions
      const nx = results.nx || 10
      const ny = results.ny || 10
      const margin = viewSize / 2
      const moveStep = 20
      
      // Draw grid
      ctx.strokeStyle = 'rgba(255, 255, 0, 0.6)'
      ctx.lineWidth = 1
      ctx.setLineDash([6, 4])
      
      for (let i = 0; i <= nx; i++) {
        const xOrig = margin + i * moveStep
        const x = xOrig * scaleX
        ctx.beginPath()
        ctx.moveTo(x, margin * scaleY)
        ctx.lineTo(x, (imgH - margin) * scaleY)
        ctx.stroke()
      }
      
      for (let j = 0; j <= ny; j++) {
        const yOrig = margin + j * moveStep
        const y = yOrig * scaleY
        ctx.beginPath()
        ctx.moveTo(margin * scaleX, y)
        ctx.lineTo((imgW - margin) * scaleX, y)
        ctx.stroke()
      }
      ctx.setLineDash([])
      
      // Draw target
      if (results.center) {
        const tx = results.center[0] * scaleX
        const ty = results.center[1] * scaleY
        ctx.strokeStyle = '#dc2626'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(tx - 10, ty)
        ctx.lineTo(tx + 10, ty)
        ctx.moveTo(tx, ty - 10)
        ctx.lineTo(tx, ty + 10)
        ctx.stroke()
        ctx.beginPath()
        ctx.arc(tx, ty, 6, 0, Math.PI * 2)
        ctx.stroke()
      }
      
      // ============ NETWORK VIEW MODE ============
      if (navViewMode === 'network') {
        // Collect all unique states and transitions from ALL trajectories
        const statePositions = new Map() // state_id -> {x, y, count}
        const transitions = new Map() // "from->to" -> count
        
        results.all_trajectories?.forEach(traj => {
          traj.steps?.forEach((step, idx) => {
            const stateId = step.state
            const pos = step.position
            
            // Aggregate positions for each state (average)
            if (!statePositions.has(stateId)) {
              statePositions.set(stateId, { x: 0, y: 0, count: 0 })
            }
            const sp = statePositions.get(stateId)
            sp.x += pos[0]
            sp.y += pos[1]
            sp.count += 1
            
            // Record transitions
            if (idx < traj.steps.length - 1) {
              const nextStep = traj.steps[idx + 1]
              const key = `${stateId}->${nextStep.state}`
              transitions.set(key, (transitions.get(key) || 0) + 1)
            }
          })
        })
        
        // Calculate average positions
        statePositions.forEach((v, k) => {
          v.x = v.x / v.count
          v.y = v.y / v.count
        })
        
        // Draw transitions as edges
        const maxCount = Math.max(...transitions.values(), 1)
        transitions.forEach((count, key) => {
          const [from, to] = key.split('->').map(Number)
          const fromPos = statePositions.get(from)
          const toPos = statePositions.get(to)
          if (!fromPos || !toPos) return
          
          const x1 = fromPos.x * scaleX
          const y1 = fromPos.y * scaleY
          const x2 = toPos.x * scaleX
          const y2 = toPos.y * scaleY
          
          // Edge thickness based on count
          const thickness = 1 + (count / maxCount) * 3
          const opacity = 0.3 + (count / maxCount) * 0.5
          
          ctx.strokeStyle = `rgba(34, 211, 238, ${opacity})`
          ctx.lineWidth = thickness
          ctx.beginPath()
          ctx.moveTo(x1, y1)
          ctx.lineTo(x2, y2)
          ctx.stroke()
          
          // Draw arrow head
          const angle = Math.atan2(y2 - y1, x2 - x1)
          const headLen = 8
          ctx.fillStyle = `rgba(34, 211, 238, ${opacity})`
          ctx.beginPath()
          ctx.moveTo(x2, y2)
          ctx.lineTo(x2 - headLen * Math.cos(angle - Math.PI / 6), y2 - headLen * Math.sin(angle - Math.PI / 6))
          ctx.lineTo(x2 - headLen * Math.cos(angle + Math.PI / 6), y2 - headLen * Math.sin(angle + Math.PI / 6))
          ctx.closePath()
          ctx.fill()
        })
        
        // Draw states as nodes
        const targetState = results.target_state
        statePositions.forEach((pos, stateId) => {
          const x = pos.x * scaleX
          const y = pos.y * scaleY
          const isTarget = stateId === targetState
          
          // Fixed node size
          const radius = isTarget ? 8 : 6
          
          // Draw node
          ctx.fillStyle = isTarget ? '#dc2626' : '#0f172a'
          ctx.strokeStyle = isTarget ? '#fca5a5' : '#22d3ee'
          ctx.lineWidth = isTarget ? 2.5 : 1.5
          
          ctx.beginPath()
          ctx.arc(x, y, radius, 0, Math.PI * 2)
          ctx.fill()
          ctx.stroke()
        })
        
        // Info overlay for network mode
        ctx.fillStyle = 'rgba(15, 23, 42, 0.85)'
        ctx.fillRect(8, 8, 180, 65)
        ctx.fillStyle = '#e2e8f0'
        ctx.font = 'bold 13px Inter, system-ui'
        ctx.textAlign = 'left'
        ctx.fillText('State Network View', 16, 28)
        ctx.font = '11px Inter, system-ui'
        ctx.fillStyle = '#94a3b8'
        ctx.fillText(`States: ${statePositions.size}  Edges: ${transitions.size}`, 16, 46)
        ctx.fillText(`From ${results.rollouts} rollouts`, 16, 62)
        
      } else {
        // ============ PLAYBACK VIEW MODE ============
        // Draw trajectory
        if (currentTrajectory.length > 0 && currentStep > 0) {
          ctx.strokeStyle = 'rgba(16, 185, 129, 0.8)'
          ctx.lineWidth = 2
          ctx.setLineDash([5, 3])
          ctx.beginPath()
          for (let i = 0; i <= currentStep && i < currentTrajectory.length; i++) {
            const step = currentTrajectory[i]
            const x = step.position[0] * scaleX
            const y = step.position[1] * scaleY
            if (i === 0) ctx.moveTo(x, y)
            else ctx.lineTo(x, y)
          }
          ctx.stroke()
          ctx.setLineDash([])
        }
        
        // Draw agent
        if (currentTrajectory.length > 0 && currentStep < currentTrajectory.length) {
          const step = currentTrajectory[currentStep]
          const x = step.position[0] * scaleX
          const y = step.position[1] * scaleY
          
          ctx.strokeStyle = '#0ea5e9'
          ctx.lineWidth = 3
          const boxW = viewSize * scaleX
          const boxH = viewSize * scaleY
          ctx.strokeRect(x - boxW/2, y - boxH/2, boxW, boxH)
          
          ctx.fillStyle = '#0ea5e9'
          ctx.beginPath()
          ctx.arc(x, y, 8, 0, Math.PI * 2)
          ctx.fill()
          ctx.strokeStyle = '#fff'
          ctx.lineWidth = 2
          ctx.stroke()
        }
        
        // Info overlay for playback mode
        ctx.fillStyle = 'rgba(15, 23, 42, 0.85)'
        ctx.fillRect(8, 8, 160, 50)
        ctx.fillStyle = '#e2e8f0'
        ctx.font = 'bold 13px Inter, system-ui'
        ctx.textAlign = 'left'
        ctx.fillText(`Rollout: ${selectedRollout + 1} / ${results.rollouts}`, 16, 28)
        ctx.fillText(`Step: ${currentStep} / ${currentTrajectory.length - 1}`, 16, 46)
        
        if (currentStep === currentTrajectory.length - 1 && results.all_trajectories?.[selectedRollout]) {
          const reached = results.all_trajectories[selectedRollout].reached_target
          ctx.fillStyle = reached ? '#10b981' : '#ef4444'
          ctx.font = 'bold 14px Inter, system-ui'
          ctx.fillText(reached ? '✓ TARGET REACHED' : '✗ NOT REACHED', 16, 80)
        }
      }
    }
    img.src = imagePreview
  }, [imagePreview, results, currentTrajectory, currentStep, selectedRollout, navViewMode])

  useEffect(() => {
    drawNavigation()
  }, [drawNavigation])

  // Playback controls
  const togglePlay = () => {
    if (currentStep >= currentTrajectory.length - 1) setCurrentStep(0)
    setIsPlaying(!isPlaying)
  }

  const resetPlayback = () => {
    setIsPlaying(false)
    setCurrentStep(0)
  }

  const prevRollout = () => {
    if (selectedRollout > 0) {
      setSelectedRollout(selectedRollout - 1)
      setCurrentStep(0)
      setIsPlaying(false)
    }
  }

  const nextRollout = () => {
    if (results && selectedRollout < results.rollouts - 1) {
      setSelectedRollout(selectedRollout + 1)
      setCurrentStep(0)
      setIsPlaying(false)
    }
  }

  // Draw network graph with hierarchical layout
  useEffect(() => {
    if (!results?.analysis?.transition_analysis || !networkRef.current || activeTab !== 'network') return
    
    const container = networkRef.current
    container.innerHTML = ''
    
    const width = container.clientWidth || 600
    const height = container.clientHeight || 700
    const padding = { top: 60, right: 80, bottom: 60, left: 80 }
    
    const graph = results.analysis.transition_analysis
    const layersData = results.analysis.layers || {}
    const targetState = results.target_state
    
    // Get current trajectory states for highlighting
    const highlightStates = new Set()
    const highlightEdges = new Set()
    if (currentTrajectory.length > 0) {
      currentTrajectory.forEach((step, idx) => {
        highlightStates.add(step.state)
        if (idx < currentTrajectory.length - 1) {
          const nextState = currentTrajectory[idx + 1].state
          highlightEdges.add(`${step.state}->${nextState}`)
        }
      })
    }
    
    // Get invariant subsets and recovery edges if not feasible
    const recoveryData = results.analysis?.recovery
    const unreachableSubsets = recoveryData?.unreachable_subsets || []
    const recoveryEdges = recoveryData?.recovery_edges || []
    
    // Create node-to-subset mapping with random colors
    const subsetColors = [
      '#f97316', '#eab308', '#22c55e', '#06b6d4', '#8b5cf6', 
      '#ec4899', '#6366f1', '#14b8a6', '#f43f5e', '#84cc16'
    ]
    const nodeToSubset = new Map()
    unreachableSubsets.forEach((subset, idx) => {
      const color = subsetColors[idx % subsetColors.length]
      subset.forEach(nodeId => {
        nodeToSubset.set(nodeId, { subsetIdx: idx, color })
      })
    })
    
    // Only include nodes that have at least one edge (incoming or outgoing)
    const nodesWithEdges = new Set()
    const links = []
    
    graph.forEach(t => {
      // Skip self-loops for display purposes
      if (t.from_state !== t.to_state) {
        nodesWithEdges.add(t.from_state)
        nodesWithEdges.add(t.to_state)
        links.push({
          source: t.from_state,
          target: t.to_state,
          robustness: t.robustness,
          is_critical: t.is_critical
        })
      }
    })
    
    // Add nodes from unreachable subsets (they might not have edges but need to be shown)
    unreachableSubsets.forEach(subset => {
      subset.forEach(nodeId => nodesWithEdges.add(nodeId))
    })
    
    // Build node-to-layer mapping
    const nodeToLayer = {}
    Object.entries(layersData).forEach(([layer, nodes]) => {
      const layerNum = parseInt(layer)
      if (layerNum >= 0) {
        nodes.forEach(nodeId => {
          if (nodesWithEdges.has(nodeId)) {
            nodeToLayer[nodeId] = layerNum
          }
        })
      }
    })
    
    // Find max layer
    const maxLayer = Math.max(...Object.values(nodeToLayer), 0)
    
    // Group nodes by layer
    const nodesByLayer = {}
    for (let i = 0; i <= maxLayer; i++) {
      nodesByLayer[i] = []
    }
    nodesWithEdges.forEach(id => {
      const layer = nodeToLayer[id] ?? maxLayer
      if (!nodesByLayer[layer]) nodesByLayer[layer] = []
      nodesByLayer[layer].push(id)
    })
    
    // Calculate positions: target (layer 0) on right, higher layers on left
    const usableWidth = width - padding.left - padding.right
    const usableHeight = height - padding.top - padding.bottom
    const layerWidth = maxLayer > 0 ? usableWidth / maxLayer : usableWidth
    
    const nodePositions = {}
    Object.entries(nodesByLayer).forEach(([layer, nodeIds]) => {
      const layerNum = parseInt(layer)
      // x: layer 0 (target) at right, higher layers at left
      const x = padding.left + usableWidth - (layerNum * layerWidth)
      
      // y: distribute nodes evenly within layer
      const nodesInLayer = nodeIds.length
      const spacing = usableHeight / (nodesInLayer + 1)
      
      nodeIds.sort((a, b) => a - b).forEach((id, idx) => {
        nodePositions[id] = {
          x,
          y: padding.top + spacing * (idx + 1)
        }
      })
    })
    
    // Create nodes with positions
    const nodes = Array.from(nodesWithEdges).map(id => ({
      id,
      isTarget: id === targetState,
      x: nodePositions[id]?.x || width / 2,
      y: nodePositions[id]?.y || height / 2,
      layer: nodeToLayer[id] ?? -1
    }))
    
    const svg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .style('cursor', 'grab')
    
    const g = svg.append('g')
    
    const zoom = d3.zoom()
      .scaleExtent([0.2, 3])
      .on('zoom', (event) => g.attr('transform', event.transform))
    
    svg.call(zoom)
    
    // Arrow markers
    const defs = svg.append('defs')
    
    defs.append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '0 -3 6 6')
      .attr('refX', 5)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 5)
      .attr('markerHeight', 5)
      .append('path')
      .attr('d', 'M 0,-2.5 L 5,0 L 0,2.5')
      .attr('fill', '#94a3b8')
    
    defs.append('marker')
      .attr('id', 'arrowhead-warning')
      .attr('viewBox', '0 -3 6 6')
      .attr('refX', 5)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 5)
      .attr('markerHeight', 5)
      .append('path')
      .attr('d', 'M 0,-2.5 L 5,0 L 0,2.5')
      .attr('fill', '#fcd34d')
    
    // Layer labels at top
    for (let i = 0; i <= maxLayer; i++) {
      const x = padding.left + usableWidth - (i * layerWidth)
      g.append('text')
        .attr('x', x)
        .attr('y', 30)
        .attr('text-anchor', 'middle')
        .attr('fill', i === 0 ? '#f87171' : '#64748b')
        .attr('font-size', '11px')
        .attr('font-weight', i === 0 ? 'bold' : 'normal')
        .text(i === 0 ? 'Target (L0)' : `L${i}`)
    }
    
    // Build lookup for positions
    const nodeById = {}
    nodes.forEach(n => nodeById[n.id] = n)
    
    // Draw curved links
    const linkGroup = g.append('g')
    
    // Add highlighted path marker
    defs.append('marker')
      .attr('id', 'arrowhead-highlight')
      .attr('viewBox', '0 -3 6 6')
      .attr('refX', 5)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .append('path')
      .attr('d', 'M 0,-2.5 L 5,0 L 0,2.5')
      .attr('fill', '#22d3ee')
    
    links.forEach(l => {
      const sourceNode = nodeById[l.source]
      const targetNode = nodeById[l.target]
      if (!sourceNode || !targetNode) return
      
      const x1 = sourceNode.x
      const y1 = sourceNode.y
      const x2 = targetNode.x
      const y2 = targetNode.y
      
      // Check if this edge is in the current trajectory (only if showPathInNetwork is true)
      const isInPath = showPathInNetwork && highlightEdges.has(`${l.source}->${l.target}`)
      
      // Calculate distance and shorten line to stop at node edge
      const dx = x2 - x1
      const dy = y2 - y1
      const dist = Math.sqrt(dx * dx + dy * dy)
      const nodeRadius = targetNode.isTarget ? 18 : 11
      
      // Shorten the end point to touch node edge (accounting for arrow size)
      const ratio = (dist - nodeRadius) / dist
      const endX = x1 + dx * ratio
      const endY = y1 + dy * ratio
      
      // Control point for curve
      const midX = (x1 + endX) / 2
      const midY = (y1 + endY) / 2 - dist * 0.12
      
      // Determine stroke color and style based on highlight and criticality
      let strokeColor = l.is_critical ? '#fcd34d' : '#64748b'
      let strokeWidth = l.is_critical ? 1.2 : 0.8
      let strokeOpacity = 0.6
      let marker = l.is_critical ? 'url(#arrowhead-warning)' : 'url(#arrowhead)'
      
      if (isInPath) {
        strokeColor = '#22d3ee' // Cyan for trajectory path
        strokeWidth = 2.5
        strokeOpacity = 1
        marker = 'url(#arrowhead-highlight)'
      }
      
      linkGroup.append('path')
        .attr('d', `M${x1},${y1} Q${midX},${midY} ${endX},${endY}`)
        .attr('fill', 'none')
        .attr('stroke', strokeColor)
        .attr('stroke-opacity', strokeOpacity)
        .attr('stroke-width', strokeWidth)
        .attr('marker-end', marker)
    })
    
    // Draw recovery edges (dashed, green) if enabled
    if (showRecoveryEdges && recoveryEdges.length > 0) {
      // Add recovery arrow marker
      defs.append('marker')
        .attr('id', 'arrowhead-recovery')
        .attr('viewBox', '0 -3 6 6')
        .attr('refX', 5)
        .attr('refY', 0)
        .attr('orient', 'auto')
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .append('path')
        .attr('d', 'M 0,-2.5 L 5,0 L 0,2.5')
        .attr('fill', '#22c55e')
      
      recoveryEdges.forEach(edge => {
        const sourceNode = nodeById[edge.source]
        const targetNode = nodeById[edge.dest]
        if (!sourceNode || !targetNode) return
        
        const x1 = sourceNode.x
        const y1 = sourceNode.y
        const x2 = targetNode.x
        const y2 = targetNode.y
        
        const dx = x2 - x1
        const dy = y2 - y1
        const dist = Math.sqrt(dx * dx + dy * dy)
        const nodeRadius = targetNode.isTarget ? 18 : 11
        
        const ratio = (dist - nodeRadius) / dist
        const endX = x1 + dx * ratio
        const endY = y1 + dy * ratio
        
        // Control point for curve (offset more for visibility)
        const midX = (x1 + endX) / 2
        const midY = (y1 + endY) / 2 - dist * 0.2
        
        linkGroup.append('path')
          .attr('d', `M${x1},${y1} Q${midX},${midY} ${endX},${endY}`)
          .attr('fill', 'none')
          .attr('stroke', '#22c55e')
          .attr('stroke-opacity', 0.9)
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', '6,3')
          .attr('marker-end', 'url(#arrowhead-recovery)')
          .style('filter', 'drop-shadow(0 0 4px rgba(34, 197, 94, 0.5))')
      })
    }
    
    // Draw nodes
    const nodeGroup = g.append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('transform', d => `translate(${d.x},${d.y})`)
      .style('cursor', 'pointer')
    
    // Node circles with gradient for target, highlighting for trajectory, and coloring for invariant subsets
    // Only show path highlighting if showPathInNetwork is true
    const showPath = showPathInNetwork
    
    nodeGroup.append('circle')
      .attr('r', d => {
        if (d.isTarget) return 18
        if (showPath && highlightStates.has(d.id)) return 13
        if (nodeToSubset.has(d.id)) return 12 // Slightly larger for invariant subset
        return 11
      })
      .attr('fill', d => {
        if (d.isTarget) return '#dc2626'
        if (showPath && highlightStates.has(d.id)) return '#0e7490' // Dark cyan for path nodes
        if (nodeToSubset.has(d.id)) {
          // Darker version of subset color for fill
          const color = nodeToSubset.get(d.id).color
          return color + '40' // With transparency
        }
        return '#0f172a'
      })
      .attr('stroke', d => {
        if (d.isTarget) return '#fca5a5'
        if (showPath && highlightStates.has(d.id)) return '#22d3ee' // Bright cyan border
        if (nodeToSubset.has(d.id)) return nodeToSubset.get(d.id).color // Subset color border
        if (d.layer === 1) return '#22c55e'
        return '#0ea5e9'
      })
      .attr('stroke-width', d => {
        if (d.isTarget) return 3
        if (showPath && highlightStates.has(d.id)) return 3
        if (nodeToSubset.has(d.id)) return 3 // Thicker border for visibility
        return 2
      })
      .attr('filter', d => {
        if (d.isTarget) return 'drop-shadow(0 0 8px rgba(239, 68, 68, 0.5))'
        if (showPath && highlightStates.has(d.id)) return 'drop-shadow(0 0 6px rgba(34, 211, 238, 0.5))'
        if (nodeToSubset.has(d.id)) return 'drop-shadow(0 0 6px rgba(249, 115, 22, 0.4))' // Orange glow
        return 'none'
      })
    
    // Node labels
    nodeGroup.append('text')
      .text(d => `S${d.id}`)
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('fill', '#fff')
      .attr('font-size', d => d.isTarget ? '10px' : '8px')
      .attr('font-weight', 'bold')
    
    // Tooltip on hover
    nodeGroup.on('mouseenter', function(event, d) {
      d3.select(this).select('circle')
        .transition().duration(150)
        .attr('r', d.isTarget ? 22 : 14)
    }).on('mouseleave', function(event, d) {
      d3.select(this).select('circle')
        .transition().duration(150)
        .attr('r', d.isTarget ? 18 : 11)
    })
    
  }, [results, activeTab, currentTrajectory, selectedRollout, showRecoveryEdges, showPathInNetwork])

  const analysis = results?.analysis
  const feasibility = analysis?.feasibility

  return (
    <div className="min-h-screen bg-slate-900 relative">
      {/* Fixed background gradient */}
      <div className="fixed inset-0 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 -z-10" />
      
      {/* Top overscroll cover */}
      <div className="fixed top-0 left-0 right-0 h-32 bg-slate-900 -translate-y-full -z-5" />
      
      {/* Header */}
      <header className="border-b border-slate-700/50 bg-slate-900/95 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center gap-4">
            <div className="p-2.5 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 shadow-lg shadow-cyan-500/20">
              <Stethoscope className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">Robotic Surgery Plan Certification</h1>
              <p className="text-slate-400 text-sm">MDP-based Analysis for Certifiable Surgical Policies</p>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto p-6">
        {/* Top Section: Input + Stats */}
        <div className="grid grid-cols-3 gap-6 mb-6">
          {/* Input Panel */}
          <div className="bg-slate-800/50 backdrop-blur rounded-2xl p-5 border border-slate-700/50">
            <div className="flex items-center gap-2 mb-4">
              <Upload className="w-5 h-5 text-cyan-400" />
              <h2 className="font-semibold text-white">Input</h2>
            </div>
            
            <div 
              className="h-32 border-2 border-dashed border-slate-600 rounded-xl flex items-center justify-center cursor-pointer hover:border-cyan-500 transition-colors overflow-hidden mb-4"
              onClick={() => document.getElementById('image-input').click()}
            >
              {imagePreview ? (
                <img src={imagePreview} alt="Preview" className="max-h-full max-w-full object-contain" />
              ) : (
                <div className="text-center text-slate-500">
                  <Upload className="w-8 h-8 mx-auto mb-1 opacity-50" />
                  <p className="text-sm">Upload ultrasound</p>
                </div>
              )}
            </div>
            <input id="image-input" type="file" accept="image/*" onChange={handleImageChange} className="hidden" />
            
            <div className="grid grid-cols-2 gap-3 mb-3">
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Rollouts</label>
                <input
                  type="number"
                  value={rolloutNum}
                  onChange={(e) => setRolloutNum(parseInt(e.target.value) || 100)}
                  disabled={loading}
                  className="w-full bg-slate-900/60 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm focus:border-cyan-500 disabled:opacity-50"
                />
              </div>
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Max Steps</label>
                <input
                  type="number"
                  value={maxSteps}
                  onChange={(e) => setMaxSteps(parseInt(e.target.value) || 50)}
                  disabled={loading}
                  className="w-full bg-slate-900/60 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm focus:border-cyan-500 disabled:opacity-50"
                />
              </div>
            </div>
            
            {/* Policy Settings */}
            <div className="mb-3 p-3 bg-slate-900/40 rounded-xl border border-slate-700/50">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Brain className="w-4 h-4 text-violet-400" />
                  <span className="text-sm text-slate-300">Policy Type</span>
                </div>
                <div className="flex items-center gap-1 bg-slate-800 rounded-lg p-0.5">
                  <button
                    onClick={() => setStochasticPolicy(true)}
                    disabled={loading}
                    className={`px-2.5 py-1 text-xs rounded-md transition-colors ${
                      stochasticPolicy 
                        ? 'bg-violet-500 text-white' 
                        : 'text-slate-400 hover:text-white'
                    } disabled:opacity-50`}
                  >
                    Stochastic
                  </button>
                  <button
                    onClick={() => setStochasticPolicy(false)}
                    disabled={loading}
                    className={`px-2.5 py-1 text-xs rounded-md transition-colors ${
                      !stochasticPolicy 
                        ? 'bg-cyan-500 text-white' 
                        : 'text-slate-400 hover:text-white'
                    } disabled:opacity-50`}
                  >
                    Argmax
                  </button>
                </div>
              </div>
              
              {/* Temperature (only for stochastic) */}
              {stochasticPolicy && (
                <div className="flex items-center gap-3">
                  <span className="text-xs text-slate-400 min-w-[70px]">Temperature</span>
                  <input
                    type="range"
                    min="0.1"
                    max="3.0"
                    step="0.1"
                    value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    disabled={loading}
                    className="flex-1 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-violet-500"
                  />
                  <span className="text-xs text-violet-400 font-mono w-8">{temperature.toFixed(1)}</span>
                </div>
              )}
              <div className="mt-1 text-xs text-slate-500">
                {stochasticPolicy 
                  ? `Softmax sampling (τ=${temperature.toFixed(1)}): different rollouts may vary`
                  : 'Argmax: all rollouts will be identical'
                }
              </div>
              
              {/* Slip Probability (Environment Stochasticity) */}
              <div className="flex items-center gap-3 mt-2">
                <span className="text-xs text-slate-400 min-w-[70px]">Slip Prob</span>
                <input
                  type="range"
                  min="0"
                  max="0.5"
                  step="0.05"
                  value={slipProb}
                  onChange={(e) => setSlipProb(parseFloat(e.target.value))}
                  disabled={loading}
                  className="flex-1 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-amber-500"
                />
                <span className="text-xs text-amber-400 font-mono w-10">{(slipProb * 100).toFixed(0)}%</span>
              </div>
              <div className="mt-1 text-xs text-slate-500">
                Environment randomness: {slipProb > 0 
                  ? `${(slipProb * 100).toFixed(0)}% chance of random action (adds graph edges)`
                  : 'Deterministic environment (sparse graph)'
                }
              </div>
            </div>
            
            {/* State Abstraction Mode */}
            <div className="mb-3 p-3 bg-slate-900/40 rounded-xl border border-slate-700/50">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Grid3X3 className="w-4 h-4 text-cyan-400" />
                  <span className="text-sm text-slate-300">State Abstraction</span>
                </div>
                <div className="flex items-center gap-1 bg-slate-800 rounded-lg p-0.5">
                  <button
                    onClick={() => setStateMode('grid')}
                    disabled={loading}
                    className={`px-2.5 py-1 text-xs rounded-md transition-colors ${
                      stateMode === 'grid' 
                        ? 'bg-cyan-500 text-white' 
                        : 'text-slate-400 hover:text-white'
                    } disabled:opacity-50`}
                  >
                    Grid
                  </button>
                  <button
                    onClick={() => setStateMode('distance')}
                    disabled={loading}
                    className={`px-2.5 py-1 text-xs rounded-md transition-colors ${
                      stateMode === 'distance' 
                        ? 'bg-orange-500 text-white' 
                        : 'text-slate-400 hover:text-white'
                    } disabled:opacity-50`}
                  >
                    View-based
                  </button>
                </div>
              </div>
              
              {/* Target Radius (only for distance mode) */}
              {stateMode === 'distance' && (
                <div className="flex items-center gap-3">
                  <span className="text-xs text-slate-400 min-w-[80px]">Target Radius</span>
                  <input
                    type="range"
                    min="5"
                    max="30"
                    step="1"
                    value={targetRadius}
                    onChange={(e) => setTargetRadius(parseInt(e.target.value))}
                    disabled={loading}
                    className="flex-1 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-orange-500"
                  />
                  <span className="text-xs text-orange-400 font-mono w-8">{targetRadius}px</span>
                </div>
              )}
              <div className="mt-1 text-xs text-slate-500">
                {stateMode === 'grid' 
                  ? 'Grid cells (20px step): target = grid cell'
                  : `Distance-based: positions within ${targetRadius}px = target state`
                }
              </div>
            </div>
            
            {/* Success Only Filter */}
            <div className="flex items-center justify-between p-3 bg-slate-900/40 rounded-xl border border-slate-700/50">
              <div className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-400" />
                <span className="text-sm text-slate-300">Analyze Success Only</span>
              </div>
              <button
                onClick={() => setSuccessOnly(!successOnly)}
                disabled={loading}
                className={`relative w-11 h-6 rounded-full transition-colors ${
                  successOnly ? 'bg-emerald-500' : 'bg-slate-600'
                } disabled:opacity-50`}
              >
                <div className={`absolute top-1 w-4 h-4 bg-white rounded-full shadow transition-transform ${
                  successOnly ? 'translate-x-6' : 'translate-x-1'
                }`} />
              </button>
            </div>
            
            <button
              onClick={runSimulation}
              disabled={loading || !imageFile}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 rounded-xl font-medium text-white transition-all shadow-lg shadow-cyan-500/20 disabled:opacity-50"
            >
              <Brain className="w-4 h-4" />
              {loading ? 'Analyzing...' : 'Run Analysis'}
            </button>
            
            {error && <div className="mt-3 p-2 bg-red-500/20 border border-red-500/50 rounded-lg text-red-300 text-xs">{error}</div>}
          </div>

          {/* Feasibility Status */}
          <div className="bg-slate-800/50 backdrop-blur rounded-2xl p-5 border border-slate-700/50">
            <div className="flex items-center gap-2 mb-4">
              <FileCheck className="w-5 h-5 text-emerald-400" />
              <h2 className="font-semibold text-white">Feasibility Analysis</h2>
            </div>
            
            {results ? (
              <div className="space-y-3">
                <div className={`p-3 rounded-xl ${analysis?.feasible ? 'bg-emerald-500/20 border border-emerald-500/50' : 'bg-red-500/20 border border-red-500/50'}`}>
                  <div className="flex items-center gap-2">
                    {analysis?.feasible ? (
                      <CheckCircle className="w-5 h-5 text-emerald-400" />
                    ) : (
                      <AlertTriangle className="w-5 h-5 text-red-400" />
                    )}
                    <span className={`font-semibold ${analysis?.feasible ? 'text-emerald-300' : 'text-red-300'}`}>
                      {analysis?.feasible ? 'FEASIBLE' : 'NOT FEASIBLE'}
                    </span>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="bg-slate-900/50 rounded-lg p-2">
                    <div className="text-slate-400 text-xs">Absorption</div>
                    <div className={`font-semibold ${feasibility?.absorption ? 'text-emerald-400' : 'text-red-400'}`}>
                      {feasibility?.absorption ? '✓ Stable' : '✗ Unstable'}
                    </div>
                  </div>
                  <div className="bg-slate-900/50 rounded-lg p-2">
                    <div className="text-slate-400 text-xs">Reachability</div>
                    <div className={`font-semibold ${feasibility?.reachability ? 'text-emerald-400' : 'text-red-400'}`}>
                      {feasibility?.reachability ? '✓ Complete' : '✗ Incomplete'}
                    </div>
                  </div>
                </div>
                
                <div className="bg-slate-900/50 rounded-lg p-2 text-sm">
                  <div className="text-slate-400 text-xs">Max Layer Depth (T_max)</div>
                  <div className="text-white font-mono">{feasibility?.max_layer ?? '-'}</div>
                </div>
                
                {/* Recovery Plan (when not feasible) */}
                {!analysis?.feasible && analysis?.recovery && (
                  <div className="mt-3 p-3 rounded-xl bg-amber-500/10 border border-amber-500/30">
                    <div className="flex items-center gap-2 mb-2">
                      <Zap className="w-4 h-4 text-amber-400" />
                      <span className="text-amber-300 font-semibold text-sm">Recovery Suggestion</span>
                    </div>
                    <div className="text-xs text-slate-300 space-y-1">
                      <div>
                        <span className="text-slate-400">Unreachable Subsets: </span>
                        <span className="text-amber-300 font-mono">{analysis.recovery.unreachable_subsets?.length || 0}</span>
                      </div>
                      <div>
                        <span className="text-slate-400">Total Unreachable: </span>
                        <span className="text-amber-300 font-mono">{analysis.recovery.total_unreachable || 0} states</span>
                      </div>
                      <div>
                        <span className="text-slate-400">Edges Needed: </span>
                        <span className="text-amber-300 font-mono">{analysis.recovery.edges_needed || 0}</span>
                      </div>
                      {analysis.recovery.recovery_edges?.length > 0 && (
                        <div className="mt-2 pt-2 border-t border-amber-500/20">
                          <div className="text-slate-400 mb-1">Suggested Edges:</div>
                          <div className="max-h-24 overflow-y-auto space-y-0.5">
                            {analysis.recovery.recovery_edges.slice(0, 5).map((edge, idx) => (
                              <div key={idx} className="font-mono text-xs">
                                <span className="text-orange-300">S{edge.source}</span>
                                <span className="text-slate-500"> → </span>
                                <span className="text-cyan-300">S{edge.dest}</span>
                                <span className="text-slate-500 ml-1">
                                  (d={edge.physical_distance?.toFixed(1) || '?'})
                                </span>
                              </div>
                            ))}
                            {analysis.recovery.recovery_edges.length > 5 && (
                              <div className="text-slate-500">+{analysis.recovery.recovery_edges.length - 5} more...</div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="h-32 flex items-center justify-center text-slate-500 text-sm">
                Run analysis to check feasibility
              </div>
            )}
          </div>

          {/* Robustness Stats */}
          <div className="bg-slate-800/50 backdrop-blur rounded-2xl p-5 border border-slate-700/50">
            <div className="flex items-center gap-2 mb-4">
              <Shield className="w-5 h-5 text-amber-400" />
              <h2 className="font-semibold text-white">Robustness Statistics</h2>
            </div>
            
            {results ? (
              <div className="space-y-3">
                {/* State Mode Indicator */}
                <div className={`flex items-center justify-center gap-2 py-1.5 px-3 rounded-lg text-xs ${
                  results.state_mode === 'distance' 
                    ? 'bg-orange-500/20 border border-orange-500/40 text-orange-300' 
                    : 'bg-cyan-500/20 border border-cyan-500/40 text-cyan-300'
                }`}>
                  <Grid3X3 className="w-3.5 h-3.5" />
                  <span className="font-medium">
                    {results.state_mode === 'distance' 
                      ? `View-based (r=${results.target_radius}px)` 
                      : 'Grid-based (20px step)'}
                  </span>
                  <span className="text-slate-400">•</span>
                  <span>{results.num_states} states</span>
                </div>
                
                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-slate-900/50 rounded-lg p-3 text-center">
                    <div className="text-2xl font-bold text-cyan-400">
                      {results.filter_applied ? results.filtered_rollout_count : results.rollouts}
                    </div>
                    <div className="text-xs text-slate-400">
                      {results.filter_applied ? 'Analyzed Rollouts' : 'Total Rollouts'}
                    </div>
                  </div>
                  <div className="bg-slate-900/50 rounded-lg p-3 text-center">
                    <div className="text-2xl font-bold text-emerald-400">
                      {results.successful_rollouts}
                      <span className="text-sm text-slate-400 ml-1">
                        ({((results.successful_rollouts / results.rollouts) * 100).toFixed(0)}%)
                      </span>
                    </div>
                    <div className="text-xs text-slate-400">Success Rate</div>
                  </div>
                </div>
                
                {/* Filter indicator */}
                {results.filter_applied && (
                  <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg p-2">
                    <div className="flex items-center justify-center gap-2 text-emerald-400 text-sm">
                      <CheckCircle className="w-4 h-4" />
                      <span>Analysis based on {results.filtered_rollout_count} successful rollouts</span>
                    </div>
                    <div className="text-xs text-slate-500 text-center mt-1">
                      ({results.rollouts - results.successful_rollouts} failed rollouts excluded)
                    </div>
                  </div>
                )}
                
                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-2 text-center">
                    <div className="text-xl font-bold text-amber-400">{analysis?.statistics?.critical_transitions ?? 0}</div>
                    <div className="text-xs text-amber-300">Critical Transitions</div>
                  </div>
                  <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg p-2 text-center">
                    <div className="text-xl font-bold text-emerald-400">{analysis?.statistics?.safe_transitions ?? 0}</div>
                    <div className="text-xs text-emerald-300">Safe Transitions</div>
                  </div>
                </div>
                
                <div className="bg-slate-900/50 rounded-lg p-2 text-sm">
                  <div className="text-slate-400 text-xs">Target State</div>
                  <div className="text-red-400 font-mono font-bold">S{results.target_state}</div>
                </div>
                
                {/* Debug info */}
                {results.debug && (
                  <div className="bg-slate-900/50 rounded-lg p-2 text-xs text-slate-500 border border-slate-700/50">
                    <div>Target row sum: {results.debug.target_row_sum}</div>
                    {results.debug.target_successors?.length > 0 && (
                      <div className="text-amber-400">⚠ Successors: {results.debug.target_successors.join(', ')}</div>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <div className="h-32 flex items-center justify-center text-slate-500 text-sm">
                Run analysis to see statistics
              </div>
            )}
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-4">
          {[
            { id: 'navigation', label: 'Navigation Playback', icon: Activity },
            { id: 'matrix', label: 'Transition Matrix', icon: Grid3X3 },
            { id: 'robustness', label: 'Robustness Analysis', icon: TrendingUp },
            { id: 'network', label: 'State Network', icon: Network },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm transition-all ${
                activeTab === tab.id
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50'
                  : 'bg-slate-800/50 text-slate-400 border border-slate-700/50 hover:border-slate-600'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Main Content Area */}
        <div className="bg-slate-800/50 backdrop-blur rounded-2xl border border-slate-700/50 min-h-[500px]">
          
          {/* Navigation Tab */}
          {activeTab === 'navigation' && (
            <div className="p-6">
              <div className="flex gap-6">
                <div className="flex-shrink-0" style={{ width: 420 }}>
                  {imagePreview ? (
                    <canvas ref={canvasRef} className="rounded-xl shadow-xl" />
                  ) : (
                    <div className="w-[420px] h-[420px] rounded-xl bg-slate-900/60 border-2 border-dashed border-slate-600 flex flex-col items-center justify-center">
                      <svg className="w-24 h-24 text-slate-600 mb-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M3 8V6a2 2 0 012-2h2M3 16v2a2 2 0 002 2h2M21 8V6a2 2 0 00-2-2h-2M21 16v2a2 2 0 01-2 2h-2" strokeLinecap="round"/>
                        <circle cx="12" cy="12" r="4" />
                        <path d="M12 8v-1M12 17v-1M8 12H7M17 12h-1" strokeLinecap="round"/>
                      </svg>
                      <p className="text-slate-500 text-sm font-medium">No Image Uploaded</p>
                      <p className="text-slate-600 text-xs mt-1">Upload an ultrasound image to begin</p>
                    </div>
                  )}
                  
                  {results && navViewMode === 'playback' && (
                    <div className="mt-4 space-y-4">
                      {/* Progress bar with step indicator */}
                      <div className="bg-slate-900/60 rounded-xl p-3">
                        <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
                          <span>Step {currentStep}</span>
                          <span>{currentTrajectory.length > 0 ? currentTrajectory.length - 1 : 0}</span>
                        </div>
                        <div className="relative">
                          {/* Background track */}
                          <div className="h-1.5 bg-slate-600/50 rounded-full" />
                          {/* Filled track */}
                          <div 
                            className="absolute left-0 top-0 h-1.5 bg-gradient-to-r from-violet-500 via-fuchsia-500 to-pink-500 rounded-full pointer-events-none"
                            style={{ width: `${currentTrajectory.length > 1 ? (currentStep / (currentTrajectory.length - 1)) * 100 : 0}%` }}
                          />
                          {/* Thumb indicator */}
                          <div 
                            className="absolute top-1/2 -translate-y-1/2 w-3.5 h-3.5 bg-white rounded-full shadow-md shadow-fuchsia-500/40 pointer-events-none"
                            style={{ left: `calc(${currentTrajectory.length > 1 ? (currentStep / (currentTrajectory.length - 1)) * 100 : 0}% - 7px)` }}
                          />
                          {/* Invisible range input for interaction */}
                          <input
                            type="range"
                            min={0}
                            max={Math.max(0, currentTrajectory.length - 1)}
                            value={currentStep}
                            onChange={(e) => { setIsPlaying(false); setCurrentStep(parseInt(e.target.value)) }}
                            className="absolute inset-0 w-full opacity-0 cursor-pointer"
                            style={{ height: 16, top: -4 }}
                          />
                        </div>
                      </div>
                      
                      {/* Playback controls */}
                      <div className="flex items-center justify-center gap-3">
                        <button onClick={togglePlay} 
                          className="flex items-center gap-2 px-5 py-2.5 bg-cyan-600 hover:bg-cyan-700 rounded-lg text-white font-medium">
                          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                          {isPlaying ? 'Pause' : 'Play'}
                        </button>
                        <button onClick={resetPlayback}
                          className="p-2.5 bg-slate-700 hover:bg-slate-600 rounded-lg text-white">
                          <RotateCcw className="w-4 h-4" />
                        </button>
                      </div>
                      
                      {/* Rollout selector */}
                      <div className="flex items-center justify-center gap-3 bg-slate-900/40 rounded-lg p-2">
                        <button onClick={prevRollout} disabled={selectedRollout === 0} 
                          className="p-1.5 bg-slate-700 hover:bg-slate-600 rounded disabled:opacity-30 text-white">
                          <ChevronLeft className="w-4 h-4" />
                        </button>
                        <div className="text-center min-w-[140px] text-sm">
                          <span className="text-slate-400">Rollout </span>
                          <span className="text-cyan-400 font-bold">{selectedRollout + 1}</span>
                          <span className="text-slate-400"> / {results.rollouts}</span>
                        </div>
                        <button onClick={nextRollout} disabled={selectedRollout >= results.rollouts - 1}
                          className="p-1.5 bg-slate-700 hover:bg-slate-600 rounded disabled:opacity-30 text-white">
                          <ChevronRight className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  )}
                  
                  {/* Network mode info */}
                  {results && navViewMode === 'network' && (
                    <div className="mt-4 bg-slate-900/60 rounded-xl p-4 border border-slate-700/50">
                      <div className="text-sm text-slate-400">
                        <p className="mb-2">This view shows the <strong className="text-violet-400">state transition network</strong> built from all {results.rollouts} rollouts:</p>
                        <ul className="list-disc list-inside space-y-1 text-xs">
                          <li>Each node represents a visited state</li>
                          <li>Node size indicates visit frequency</li>
                          <li>Edge thickness indicates transition frequency</li>
                          <li>Arrows show transition direction</li>
                        </ul>
                      </div>
                    </div>
                  )}
                </div>
                
                <div className="flex-1 text-slate-300">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-lg font-semibold text-white">Navigation Visualization</h3>
                    {/* View Mode Toggle */}
                    <div className="flex bg-slate-700/50 rounded-lg p-0.5">
                      <button
                        onClick={() => setNavViewMode('playback')}
                        className={`px-3 py-1 text-xs rounded-md transition-all ${
                          navViewMode === 'playback'
                            ? 'bg-cyan-500 text-white shadow-md'
                            : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        Playback
                      </button>
                      <button
                        onClick={() => setNavViewMode('network')}
                        className={`px-3 py-1 text-xs rounded-md transition-all ${
                          navViewMode === 'network'
                            ? 'bg-violet-500 text-white shadow-md'
                            : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        Network
                      </button>
                    </div>
                  </div>
                  <p className="text-sm text-slate-400 mb-3">
                    {navViewMode === 'playback' 
                      ? 'Agent navigates to target. Blue box = field of view, red crosshair = target.'
                      : 'All visited states and transitions from all rollouts overlaid on image.'}
                  </p>
                  
                  <div className="flex flex-wrap gap-3 text-xs mb-4">
                    {navViewMode === 'playback' ? (
                      <>
                        <div className="flex items-center gap-1.5">
                          <div className="w-2.5 h-2.5 rounded-full bg-cyan-500 border border-white"></div>
                          <span>Agent</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <div className="text-red-500 text-xs font-bold">+</div>
                          <span>Target</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <div className="w-3 h-2 border border-cyan-500"></div>
                          <span>View</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <div className="w-4 border-t border-dashed border-yellow-400"></div>
                          <span>Grid</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <div className="w-4 border-t-2 border-dashed border-emerald-400"></div>
                          <span>Path</span>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="flex items-center gap-1.5">
                          <div className="w-2.5 h-2.5 rounded-full bg-slate-800 border-2 border-cyan-400"></div>
                          <span>State Node</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <div className="w-2.5 h-2.5 rounded-full bg-red-600 border-2 border-red-300"></div>
                          <span>Target State</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <div className="w-4 h-0.5 bg-cyan-400"></div>
                          <span>Transition</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <div className="w-4 border-t border-dashed border-yellow-400"></div>
                          <span>Grid</span>
                        </div>
                      </>
                    )}
                  </div>
                  
                  {/* State Path Display (only in playback mode) */}
                  {navViewMode === 'playback' && results && currentTrajectory.length > 0 && (
                    <div className="bg-slate-900/60 rounded-xl p-4 border border-slate-700/50">
                      <div className="flex items-center gap-2 mb-3">
                        <Target className="w-4 h-4 text-cyan-400" />
                        <span className="text-sm font-semibold text-white">State Transition Path</span>
                      </div>
                      
                      <div className="flex flex-wrap items-center gap-1 font-mono text-xs">
                        {currentTrajectory.slice(0, currentStep + 1).map((step, idx) => (
                          <span key={idx} className="flex items-center">
                            <span className={`px-1.5 py-0.5 rounded ${
                              idx === currentStep 
                                ? 'bg-cyan-500 text-white font-bold' 
                                : step.state === results.target_state 
                                  ? 'bg-red-500/30 text-red-300'
                                  : 'bg-slate-700 text-slate-300'
                            }`}>
                              S{step.state}
                            </span>
                            {idx < currentStep && (
                              <span className="text-slate-500 mx-0.5">→</span>
                            )}
                          </span>
                        ))}
                        {currentStep < currentTrajectory.length - 1 && (
                          <span className="text-slate-500">...</span>
                        )}
                      </div>
                      
                      <div className="mt-3 pt-3 border-t border-slate-700/50 flex items-center justify-between text-xs">
                        <div>
                          <span className="text-slate-400">Current: </span>
                          <span className="text-cyan-400 font-bold">S{currentTrajectory[currentStep]?.state}</span>
                        </div>
                        <div>
                          <span className="text-slate-400">Target: </span>
                          <span className="text-red-400 font-bold">S{results.target_state}</span>
                        </div>
                        <div>
                          <span className="text-slate-400">Length: </span>
                          <span className="text-white font-bold">{currentTrajectory.length}</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Transition Matrix Tab */}
          {activeTab === 'matrix' && (
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white">Transition Probability Matrix P<sub>π</sub></h3>
                {results && (
                  <div className="text-sm text-slate-400">
                    Target State: <span className="text-red-400 font-mono font-bold">S{results.target_state}</span>
                  </div>
                )}
              </div>
              {results?.analysis?.transition_probability_matrix ? (
                <div className="overflow-auto max-h-[400px]">
                  <table className="text-xs font-mono">
                    <thead>
                      <tr>
                        <th className="p-1 text-slate-400 sticky top-0 bg-slate-800 z-10">→</th>
                        {results.analysis.transition_probability_matrix[0]?.map((_, j) => (
                          <th key={j} className={`p-1 sticky top-0 bg-slate-800 min-w-[40px] ${j === results.target_state ? 'text-red-400' : 'text-slate-400'}`}>
                            S{j}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {results.analysis.transition_probability_matrix.map((row, i) => {
                        const isTargetRow = i === results.target_state
                        return (
                          <tr key={i} className={isTargetRow ? 'bg-red-500/10' : ''}>
                            <td className={`p-1 font-semibold sticky left-0 z-10 ${isTargetRow ? 'bg-red-500/20 text-red-400' : 'bg-slate-800 text-slate-400'}`}>
                              S{i}
                            </td>
                            {row.map((val, j) => (
                              <td 
                                key={j} 
                                className={`p-1 text-center ${
                                  isTargetRow ? 'text-red-300/50' :
                                  val > 0.5 ? 'bg-cyan-500/40 text-cyan-200' :
                                  val > 0.1 ? 'bg-cyan-500/20 text-cyan-300' :
                                  val > 0 ? 'bg-slate-700/50 text-slate-300' : 'text-slate-600'
                                }`}
                              >
                                {isTargetRow ? (i === j ? '1.00' : '-') : (val > 0 ? val.toFixed(2) : '-')}
                              </td>
                            ))}
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="h-64 flex items-center justify-center text-slate-500">
                  Run analysis to see transition matrix
                </div>
              )}
            </div>
          )}

          {/* Robustness Analysis Tab */}
          {activeTab === 'robustness' && (
            <div className="p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Transition Robustness Analysis ρ(i,j)</h3>
              {results?.analysis?.transition_analysis ? (
                <div className="overflow-auto max-h-[400px]">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-slate-800">
                      <tr className="text-left text-slate-400">
                        <th className="p-3">Transition</th>
                        <th className="p-3">Invariant Subset Ω(i,j)</th>
                        <th className="p-3">Size</th>
                        <th className="p-3">Robustness ρ(i,j)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.analysis.transition_analysis.map((t, idx) => (
                        <tr 
                          key={idx}
                          className={`border-b border-slate-700/50 ${
                            t.is_critical ? 'bg-amber-500/10' : ''
                          }`}
                        >
                          <td className="p-3 font-mono text-slate-200">
                            S{t.from_state} → S{t.to_state}
                          </td>
                          <td className="p-3 font-mono">
                            {t.invariant_size > 0 ? (
                              <span className="text-amber-400">
                                {'{' + t.invariant_subset.slice(0, 8).join(', ') + (t.invariant_subset.length > 8 ? ', ...' : '') + '}'}
                              </span>
                            ) : (
                              <span className="text-slate-500">∅</span>
                            )}
                          </td>
                          <td className="p-3">
                            <span className={t.invariant_size > 0 ? 'text-amber-400' : 'text-slate-500'}>
                              {t.invariant_size}
                            </span>
                          </td>
                          <td className="p-3">
                            <div className="flex items-center gap-2">
                              <div className="w-24 h-2 bg-slate-700 rounded-full overflow-hidden">
                                <div
                                  className={`h-full rounded-full ${
                                    t.robustness >= 0.95 ? 'bg-emerald-500' :
                                    t.robustness >= 0.8 ? 'bg-amber-500' : 'bg-orange-500'
                                  }`}
                                  style={{ width: `${t.robustness * 100}%` }}
                                />
                              </div>
                              <span className={`font-mono text-sm ${
                                t.robustness >= 0.95 ? 'text-emerald-400' :
                                t.robustness >= 0.8 ? 'text-amber-400' : 'text-orange-400'
                              }`}>
                                {t.robustness.toFixed(3)}
                              </span>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="h-64 flex items-center justify-center text-slate-500">
                  Run analysis to see robustness data
                </div>
              )}
            </div>
          )}

          {/* Network Tab */}
          {activeTab === 'network' && (
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-4">
                  <h3 className="text-lg font-semibold text-white">State Transition Network</h3>
                  {results?.analysis?.transition_analysis && (
                    <div className="flex items-center gap-2 text-xs">
                      <div className="bg-slate-900/60 px-2 py-1 rounded-lg">
                        <span className="text-slate-400">Nodes: </span>
                        <span className="text-cyan-400 font-bold">{networkStats.nodeCount}</span>
                      </div>
                      <div className="bg-slate-900/60 px-2 py-1 rounded-lg">
                        <span className="text-slate-400">Edges: </span>
                        <span className="text-cyan-400 font-bold">{networkStats.edgeCount}</span>
                      </div>
                      <div className="bg-violet-500/10 px-2 py-1 rounded-lg border border-violet-500/30">
                        <span className="text-slate-400">Layers: </span>
                        <span className="text-violet-400 font-bold">{results.analysis.statistics?.max_layer || 0}</span>
                      </div>
                      <div className="bg-amber-500/10 px-2 py-1 rounded-lg border border-amber-500/30">
                        <span className="text-slate-400">Critical: </span>
                        <span className="text-amber-400 font-bold">{networkStats.criticalCount}</span>
                      </div>
                    </div>
                  )}
                  {/* Rollout selector for path highlighting */}
                  {results?.all_trajectories?.length > 0 && (
                    <div className="flex items-center gap-2 bg-cyan-500/10 px-3 py-1.5 rounded-lg border border-cyan-500/30">
                      <span className="text-xs text-cyan-300">Show Path:</span>
                      <button onClick={prevRollout} disabled={selectedRollout === 0} 
                        className="p-0.5 hover:bg-slate-700 rounded disabled:opacity-30 text-white">
                        <ChevronLeft className="w-3.5 h-3.5" />
                      </button>
                      <span className="text-xs font-mono text-cyan-400 min-w-[60px] text-center">
                        Rollout {selectedRollout + 1}
                      </span>
                      <button onClick={nextRollout} disabled={selectedRollout >= results.rollouts - 1}
                        className="p-0.5 hover:bg-slate-700 rounded disabled:opacity-30 text-white">
                        <ChevronRight className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  )}
                </div>
              </div>
              
              {/* Legend and Controls Row */}
              <div className="flex items-start justify-between mb-4 gap-4">
                {/* Legend Box */}
                <div className="bg-slate-900/60 rounded-xl p-3 border border-slate-700/50">
                  <div className="text-xs text-slate-500 mb-2 font-medium">Legend</div>
                  <div className="grid grid-cols-3 gap-x-4 gap-y-1.5 text-xs text-slate-400">
                    {/* Nodes */}
                    <div className="flex items-center gap-1.5">
                      <div className="w-3 h-3 rounded-full bg-red-600 border-2 border-red-300"></div>
                      <span>Target</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <div className="w-2.5 h-2.5 rounded-full bg-slate-900 border-2 border-green-500"></div>
                      <span>L1 (1 step)</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <div className="w-2.5 h-2.5 rounded-full bg-slate-900 border-2 border-cyan-500"></div>
                      <span>Other</span>
                    </div>
                    {/* Edges */}
                    <div className="flex items-center gap-1.5">
                      <div className="w-4 h-0.5 bg-slate-500"></div>
                      <span>Normal Edge</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <div className="w-4 h-0.5 bg-amber-300"></div>
                      <span>Critical Edge</span>
                    </div>
                    <button 
                      onClick={() => setShowPathInNetwork(!showPathInNetwork)}
                      className={`flex items-center gap-1.5 px-2 py-1 rounded-md transition-all ${
                        showPathInNetwork 
                          ? 'bg-cyan-900/50 ring-1 ring-cyan-500' 
                          : 'hover:bg-slate-700/50'
                      }`}
                    >
                      <div className={`w-2.5 h-2.5 rounded-full transition-all ${
                        showPathInNetwork 
                          ? 'bg-cyan-700 border-2 border-cyan-400' 
                          : 'bg-slate-600 border-2 border-slate-500'
                      }`}></div>
                      <span className={showPathInNetwork ? 'text-cyan-400' : 'text-slate-500'}>
                        {showPathInNetwork ? 'Path (On)' : 'Path (Off)'}
                      </span>
                    </button>
                    {/* Recovery (conditional) */}
                    {results?.analysis?.recovery?.unreachable_subsets?.length > 0 && (
                      <>
                        <div className="flex items-center gap-1.5">
                          <div className="w-2.5 h-2.5 rounded-full border-2 border-orange-500" style={{backgroundColor: 'rgba(249,115,22,0.3)'}}></div>
                          <span className="text-orange-400">Unreachable</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <div className="w-4 border-t-2 border-dashed border-green-500"></div>
                          <span className="text-green-400">Recovery</span>
                        </div>
                      </>
                    )}
                  </div>
                </div>
                
                {/* Recovery Controls (if applicable) */}
                {results?.analysis?.recovery?.recovery_edges?.length > 0 && (
                  <div className="bg-slate-900/60 rounded-xl p-3 border border-slate-700/50 flex-1 max-w-md">
                    <div className="text-xs text-slate-500 mb-2 font-medium">Recovery Plan</div>
                    <div className="flex items-center gap-3">
                      <button
                        onClick={() => setShowRecoveryEdges(!showRecoveryEdges)}
                        className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                          showRecoveryEdges 
                            ? 'bg-green-500 text-white shadow-lg shadow-green-500/30' 
                            : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                        }`}
                      >
                        <Zap className="w-3 h-3" />
                        {showRecoveryEdges ? 'Hide' : 'Show'} Edges
                        <span className="px-1 py-0.5 rounded bg-white/20 text-[10px]">
                          {results.analysis.recovery.recovery_edges.length}
                        </span>
                      </button>
                      
                      <div className="flex items-center gap-1.5 text-xs">
                        <span className="text-slate-400">Subsets:</span>
                        {results.analysis.recovery.unreachable_subsets.map((subset, idx) => (
                          <span 
                            key={idx} 
                            className="px-1.5 py-0.5 rounded text-white text-[10px] font-mono"
                            style={{
                              backgroundColor: ['#f97316', '#eab308', '#22c55e', '#06b6d4', '#8b5cf6', '#ec4899'][idx % 6] + '60',
                              border: `1px solid ${['#f97316', '#eab308', '#22c55e', '#06b6d4', '#8b5cf6', '#ec4899'][idx % 6]}`
                            }}
                          >
                            {subset.length}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
              
              {results ? (
                <div ref={networkRef} className="w-full h-[700px] bg-slate-900/50 rounded-xl" />
              ) : (
                <div className="h-[700px] flex items-center justify-center text-slate-500">
                  <div className="text-center">
                    <Network className="w-16 h-16 mx-auto mb-4 opacity-50" />
                    <p>Run analysis to see network</p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      
      {/* Bottom overscroll cover */}
      <div className="h-32 bg-slate-900" />
    </div>
  )
}

export default App
