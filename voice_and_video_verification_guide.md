# Voice & Video Verification System: Complete Guide 🚀

Yeh document hamare system ke kaam karne ke tarike aur aage ke video verification plan ko aasan Hinglish me samjhane ke liye hai.

---

## Part 1: Current Voice Verification (Jo abhi chal raha hai) 🎙️

Abhi hamara system voice notes ko fully verify karta hai. Jab bhi koi audio file aati hai, hamara Python API usko **3 stages** se guzarta hai:

### 1. Replay Attack Detection (Anti-Spoofing) 📱
* **Kyu zaroori hai?** Agar koi ladka kisi ladki ki recording apne dusre mobile me play karke mic ke paas rakh de, toh use pakadne ke liye.
* **Kaise kaam karta hai?** Jab audio speaker se play hoti hai, toh uske high-frequency (4-8kHz) sounds dab jaate hain. Hamara code high-to-low energy ratio check karke use turant block kar deta hai aur report me `🔴 Replay Attack` show karta hai.

### 2. AI Voice & Deepfake Detection 🤖
* **Kyu zaroori hai?** Aaj kal AI se synthetic voice banana bohot aasan hai.
* **Kaise kaam karta hai?** Hum Hugging Face ka pre-trained deepfake detector AI model use karte hain. Yeh model aawaz ki texture check karke batata hai ki aawaz real human ki hai ya AI generator se bani hai.

### 3. Voice Gender Classification (SVM, Gradient Boost, Random Forest) 👩/👨
* **Kaise kaam karta hai?** Hum aawaz ki pitch, frequency, aur tone extract karte hain. Phir humare 3 machine learning models check karte hain ki aawaz male ki hai ya female ki.
* **Strict Rule:** Sir ki instruction ke mutabik, agar koi female voice reject bhi ho jaye toh chalega, par koi male accept nahi hona chahiye. Isliye humne rules bohot strict rakhe hain. Borderline cases ko hum **Manual Review** me bhej dete hain.

---

## Part 2: n8n Workflow & Database Routing ⚙️

Hamare n8n workflow me ek **Switch** aur ek **Error Trigger** set kiya gaya hai:

1. **Webhook:** Mobile app se voice ka URL leta hai aur Python API ko bhejta hai.
2. **Switch Node (Checks):**
   * **Approved (Female):** Agar aawaz female aur real hai, toh data seedha AWS MySQL Database me chala jata hai.
   * **Manual Review:** Agar aawaz borderline hai, toh data hold ho jata hai aur Manager ko **Email + Telegram** chala jata hai manual verification ke liye.
   * **Reject (Male / AI / Spoof):** Agar voice male, AI ya speaker playback hai, toh workflow use direct reject kar deta hai.
3. **Error Trigger Node:** Agar MySQL database down ho jaye ya API me koi error aaye, toh admin ko turant **Error Alert Email** chala jata.

---

## Part 3: Proposed Video Verification Plan (Aage kya karenge) 📹

Jab hum video verification shuru karenge, toh system aur bhi zyada secure ho jayega. Usme hum **Multimodal Verification** lagayenge:

1. **Audio Extraction:** Video se aawaz alag karke hamare current voice filter (Replay + AI + Female check) se gujarenge.
2. **Visual Gender Check (Face):** AI video ke frames (photos) ko scan karega aur confirm karega ki screen par jo insaan hai wo sach me **female** hi hai.
3. **AI Video / Deepfake Check:** Pre-trained model se check karenge ki video real camera se record hui hai ya kisi computer software (AI video generator/Face swap) se bani hai.
4. **Low Quality Handling:**
   * Agar video blurry ya dark hai aur face detect nahi ho raha, toh user ko automatic message jayega: *"Face not visible. Please upload a clear video."*
   * Agar video quality ki wajah se AI 100% sure nahi hai, toh wo auto-reject karne ke bajaye **Manual Review** me bhej dega.
