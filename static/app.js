const fileInput=document.getElementById('fileInput');
const browseBtn=document.getElementById('browseBtn');
const dropzone=document.getElementById('dropzone');
const previewWrap=document.getElementById('previewWrap');
const preview=document.getElementById('preview');
const analyzeBtn=document.getElementById('analyzeBtn');
const removeImage=document.getElementById('removeImage');
const resultBox=document.getElementById('resultBox');
const historyList=document.getElementById('historyList');
const emptyHistory=document.getElementById('emptyHistory');
const historyCount=document.getElementById('historyCount');
let selectedFile=null, scans=0;
document.getElementById('today').textContent=new Intl.DateTimeFormat('en-US',{weekday:'short',month:'short',day:'numeric',year:'numeric'}).format(new Date());
document.getElementById('menuBtn').addEventListener('click',()=>document.getElementById('sidebar').classList.toggle('open'));
browseBtn.addEventListener('click',()=>fileInput.click());
dropzone.addEventListener('click',e=>{if(e.target===dropzone||e.target.closest('.upload-icon')||e.target.closest('h3')||e.target.closest('p'))fileInput.click()});
fileInput.addEventListener('change',()=>setFile(fileInput.files[0]));
['dragenter','dragover'].forEach(type=>dropzone.addEventListener(type,e=>{e.preventDefault();dropzone.classList.add('dragover')}));
['dragleave','drop'].forEach(type=>dropzone.addEventListener(type,e=>{e.preventDefault();dropzone.classList.remove('dragover')}));
dropzone.addEventListener('drop',e=>{const f=e.dataTransfer.files[0];setFile(f)});
function setFile(file){
  if(!file)return;
  if(!['image/jpeg','image/png','image/webp'].includes(file.type)){showError('Please choose a JPG, PNG or WEBP image.');return}
  if(file.size>8*1024*1024){showError('Image must be smaller than 8 MB.');return}
  selectedFile=file;preview.src=URL.createObjectURL(file);previewWrap.hidden=false;
  [...dropzone.children].forEach(el=>{if(el!==previewWrap)el.hidden=true});
  analyzeBtn.disabled=false;resultBox.hidden=true;
}
function clearFile(){selectedFile=null;fileInput.value='';preview.src='';previewWrap.hidden=true;[...dropzone.children].forEach(el=>el.hidden=false);analyzeBtn.disabled=true}
removeImage.addEventListener('click',e=>{e.stopPropagation();clearFile()});
function showError(message){resultBox.hidden=false;resultBox.innerHTML=`<p class="error-message">${escapeHtml(message)}</p>`}
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
analyzeBtn.addEventListener('click',async()=>{
  if(!selectedFile)return;
  analyzeBtn.disabled=true;analyzeBtn.textContent='Analyzing…';dropzone.classList.add('loading');
  resultBox.hidden=false;resultBox.innerHTML='<p>Analyzing image with the configured AI model…</p>';
  try{
    const data=new FormData();data.append('image',selectedFile);
    const res=await fetch('/api/analyze',{method:'POST',body:data});
    const result=await res.json();
    if(!res.ok)throw new Error(result.error||'Analysis failed.');
    showResult(result);
  }catch(err){showError(err.message||'Could not connect to the server.')}
  finally{analyzeBtn.disabled=!selectedFile;analyzeBtn.innerHTML='Analyze leaf <span>↗</span>';dropzone.classList.remove('loading')}
});
function showResult(r){
  resultBox.hidden=false;
  resultBox.innerHTML=`<div class="result-title"><img src="${escapeHtml(r.image_url)}" alt="Analyzed leaf"><div><div class="section-kicker">ANALYSIS RESULT</div><h3>${escapeHtml(r.disease)}</h3><div class="result-meta">${escapeHtml(r.crop)} · ${escapeHtml(r.severity)} severity</div></div><div class="confidence">${escapeHtml(r.confidence)}%</div></div><h4>What the model observed</h4><p>${escapeHtml(r.symptoms)}</p><h4>Suggested next steps</h4><ul>${r.actions.map(a=>`<li>${escapeHtml(a)}</li>`).join('')}</ul><p class="result-disclaimer">${escapeHtml(r.disclaimer)}</p>`;
  scans++;historyCount.textContent=`${scans} scan${scans===1?'':'s'}`;emptyHistory.hidden=true;
  const row=document.createElement('div');row.className='history-row';
  row.innerHTML=`<img src="${escapeHtml(r.image_url)}" alt=""><div><strong>${escapeHtml(r.disease)}</strong><small>${escapeHtml(r.crop)} · ${escapeHtml(r.timestamp)}</small></div><span class="history-badge">${escapeHtml(r.confidence)}%</span>`;
  historyList.prepend(row);document.getElementById('history').scrollIntoView({behavior:'smooth',block:'nearest'});
}
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatMessages = document.getElementById("chatMessages");

if (chatForm && chatInput && chatMessages) {
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const message = chatInput.value.trim();
    if (!message) return;

    chatMessages.insertAdjacentHTML(
      "beforeend",
      `<p><strong>You:</strong> ${escapeHtml(message)}</p>`
    );
    chatInput.value = "";

    const waiting = document.createElement("p");
    waiting.textContent = "Crop Doctor is thinking...";
    chatMessages.appendChild(waiting);

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message })
      });

      const data = await response.json();
      waiting.textContent =
        "Crop Doctor: " + (data.reply || data.error || "Please try again.");
    } catch (error) {
      waiting.textContent = "Server se connection nahi ho paya.";
    }
  });
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}