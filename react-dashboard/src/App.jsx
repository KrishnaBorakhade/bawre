import { useEffect, useState } from 'react';
import { collection, query, orderBy, limit, onSnapshot } from 'firebase/firestore';
import { ref, listAll, getDownloadURL } from 'firebase/storage';
import { db, storage } from './firebase';
import { Activity, Clock, FileVideo, ActivitySquare, ShieldCheck, ShieldAlert } from 'lucide-react';
import './index.css';

function App() {
  const [events, setEvents] = useState([]);
  const [graphs, setGraphs] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // 1. Listen to real-time updates from Firestore
    const q = query(collection(db, 'medication_events'), orderBy('timestamp', 'desc'), limit(50));
    
    const unsubscribe = onSnapshot(q, (snapshot) => {
      const eventData = snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }));
      setEvents(eventData);
      setLoading(false);
    }, (error) => {
      console.error("Error fetching events:", error);
      setLoading(false);
    });

    // 2. Fetch Graph URLs from Storage
    const fetchGraphs = async () => {
      try {
        const listRef = ref(storage, 'graphs/');
        const res = await listAll(listRef);
        
        // Get top 4 most recent graphs (assuming they are named with timestamps)
        const sortedItems = res.items.sort((a, b) => b.name.localeCompare(a.name)).slice(0, 4);
        
        const urlPromises = sortedItems.map(itemRef => getDownloadURL(itemRef));
        const urls = await Promise.all(urlPromises);
        setGraphs(urls);
      } catch (error) {
        console.error("Error fetching graphs:", error);
      }
    };

    fetchGraphs();

    return () => unsubscribe();
  }, []);

  const totalEvents = events.length;
  const successEvents = events.filter(e => e.success || e.confidence_score >= 70).length;
  const complianceRate = totalEvents > 0 ? Math.round((successEvents / totalEvents) * 100) : 0;

  return (
    <div className="dashboard-container">
      <header className="header">
        <h1><ActivitySquare size={32} /> Alzheimer Medication Dashboard</h1>
        <div>Live Monitoring Panel</div>
      </header>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-title">Total Monitored Events</div>
          <div className="stat-value">{loading ? "..." : totalEvents}</div>
        </div>
        <div className="stat-card">
          <div className="stat-title">Successful Doses</div>
          <div className="stat-value" style={{ color: "var(--success)" }}>
            {loading ? "..." : successEvents}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-title">Overall Compliance</div>
          <div className="stat-value" style={{ color: complianceRate >= 80 ? "var(--success)" : "var(--warning)" }}>
            {loading ? "..." : `${complianceRate}%`}
          </div>
        </div>
      </div>

      <div className="main-grid">
        <div className="section-card">
          <h2 className="section-header"><Clock size={24} /> Medication History Timeline</h2>
          {loading ? (
            <p>Loading records...</p>
          ) : events.length === 0 ? (
            <p>No medication events recorded yet.</p>
          ) : (
            <div className="timeline">
              {events.map((ev) => {
                const date = new Date(ev.timestamp);
                const isSuccess = ev.success || ev.confidence_score >= 70;
                
                return (
                  <div key={ev.id} className={`event-card ${isSuccess ? 'success' : 'missed'}`}>
                    <div className="event-header">
                      <div className="event-action">
                        {isSuccess ? <ShieldCheck size={18} color="var(--success)" className="inline" /> : <ShieldAlert size={18} color="var(--danger)" className="inline" /> }
                        {' '}{ev.action_detected}
                      </div>
                      <div className="event-time">
                        {date.toLocaleDateString()} {date.toLocaleTimeString()}
                      </div>
                    </div>
                    
                    <div className="event-details">
                      <span>Confidence: {ev.confidence_score}%</span>
                      <span>Duration: {ev.duration_seconds}s</span>
                      <span className={`badge ${isSuccess ? 'badge-success' : 'badge-danger'}`}>
                        {isSuccess ? 'Verified' : 'Uncertain'}
                      </span>
                    </div>

                    {ev.video_url && (
                      <a href={ev.video_url} target="_blank" rel="noopener noreferrer" className="btn-video">
                        <FileVideo size={16} /> View Recording
                      </a>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div className="section-card">
          <h2 className="section-header"><Activity size={24} /> Analytics & Graphs</h2>
          {graphs.length === 0 ? (
            <p>No analytics graphs generated yet.</p>
          ) : (
            <div className="graphs-grid">
              {graphs.map((url, index) => (
                <div key={index} className="graph-item">
                  <img src={url} alt={`Analytics graph ${index}`} loading="lazy" />
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
