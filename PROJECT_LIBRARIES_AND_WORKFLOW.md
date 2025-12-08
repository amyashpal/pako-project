# Gujarat Crop Recommendation System - Libraries & Workflow Documentation

## 📚 Complete Library Usage Guide

### 1. **Frontend & Web Framework**

#### **Streamlit** (`streamlit`)
- **Purpose**: Main web application framework
- **Usage**: 
  - Creates interactive web interface
  - Handles user input forms and buttons
  - Displays results and visualizations
  - Manages page routing and navigation
- **Key Features Used**:
  - `st.title()`, `st.header()` - Page titles
  - `st.button()`, `st.selectbox()` - User inputs
  - `st.columns()` - Layout management
  - `st.session_state` - State management
  - `st.rerun()` - Page refresh

#### **extra-streamlit-components** (`extra_streamlit_components`)
- **Purpose**: Additional Streamlit widgets
- **Usage**: Enhanced UI components not in core Streamlit
- **Features**: Custom buttons, navigation bars

#### **streamlit-cookies-manager** (`streamlit_cookies_manager`)
- **Purpose**: Browser cookie management
- **Usage**:
  - Store user authentication tokens
  - Remember user login sessions
  - Persist user preferences
- **Implementation**: `CookieManager(prefix="myapp/")`

---

### 2. **Database & Data Persistence**

#### **psycopg2** (`psycopg2`)
- **Purpose**: PostgreSQL database adapter
- **Usage**:
  - Connect to PostgreSQL database
  - Execute SQL queries
  - Manage user data, soil details, predictions
- **Key Operations**:
  ```python
  conn = psycopg2.connect(**db_params)
  cur = conn.cursor()
  cur.execute("SELECT * FROM users WHERE email = %s", (email,))
  ```

#### **python-dotenv** (`dotenv`)
- **Purpose**: Environment variable management
- **Usage**:
  - Load database credentials from .env file
  - Manage API keys and secrets
  - Keep sensitive data secure
- **Implementation**: `load_dotenv()` loads `.env` file

---

### 3. **Authentication & Security**

#### **bcrypt** (`bcrypt`)
- **Purpose**: Password hashing and verification
- **Usage**:
  - Hash passwords before storing in database
  - Verify passwords during login
  - Secure password storage
- **Key Functions**:
  ```python
  # Hash password
  hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
  # Verify password
  bcrypt.checkpw(password.encode(), stored_hash.encode())
  ```

#### **PyJWT** (`jwt`)
- **Purpose**: JSON Web Token generation and verification
- **Usage**:
  - Create authentication tokens
  - Verify user sessions
  - Secure API endpoints
- **Implementation**:
  ```python
  token = jwt.encode({'user_id': user_id, 'exp': expiry}, secret_key)
  payload = jwt.decode(token, secret_key, algorithms=['HS256'])
  ```

---

### 4. **Machine Learning & Data Science**

#### **pandas** (`pandas`)
- **Purpose**: Data manipulation and analysis
- **Usage**:
  - Load and process crop dataset (CSV)
  - Data cleaning and preprocessing
  - Feature engineering
  - Data transformation
- **Key Operations**:
  ```python
  df = pd.read_csv('gujarat_full_crop_dataset.csv')
  df['NPK_ratio'] = df['Nitrogen'] / (df['Phosphorus'] + 1)
  ```

#### **numpy** (`numpy`)
- **Purpose**: Numerical computing
- **Usage**:
  - Array operations
  - Mathematical calculations
  - Feature scaling and normalization
- **Examples**:
  ```python
  np.sin(2 * np.pi * df['Month'] / 12)  # Cyclical encoding
  np.clip(df['Soil_pH'], 4.0, 9.5)  # Value clipping
  ```

#### **scikit-learn** (`sklearn`)
- **Purpose**: Machine learning algorithms and tools
- **Usage**:
  - Model training and evaluation
  - Data preprocessing
  - Feature scaling
  - Cross-validation
- **Components Used**:
  - `train_test_split` - Split data into train/test sets
  - `StandardScaler` - Feature normalization
  - `LabelEncoder` - Encode categorical variables
  - `RandomForestClassifier` - Suitability prediction
  - `accuracy_score`, `f1_score` - Model evaluation
  - `confusion_matrix` - Performance analysis

#### **XGBoost** (`xgboost`)
- **Purpose**: Gradient boosting machine learning
- **Usage**:
  - Crop recommendation (multi-class classification)
  - Yield prediction (regression)
  - High-performance predictions
- **Models**:
  ```python
  # Crop classification
  xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.03)
  # Yield prediction
  xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.02)
  ```

#### **imbalanced-learn** (`imblearn`)
- **Purpose**: Handle imbalanced datasets
- **Usage**:
  - SMOTE (Synthetic Minority Oversampling Technique)
  - Balance crop classes in training data
  - Improve minority class predictions
- **Implementation**:
  ```python
  from imblearn.over_sampling import SMOTE
  smote = SMOTE(random_state=42, k_neighbors=5)
  X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
  ```

#### **matplotlib** (`matplotlib`)
- **Purpose**: Data visualization
- **Usage**:
  - Create plots and charts
  - Visualize model performance
  - Generate confusion matrices
- **Examples**: Confusion matrix plots, feature importance charts

#### **seaborn** (`seaborn`)
- **Purpose**: Statistical data visualization
- **Usage**:
  - Enhanced matplotlib plots
  - Heatmaps for confusion matrices
  - Distribution plots
- **Implementation**: `sns.heatmap(confusion_matrix, annot=True)`

#### **SHAP** (`shap`)
- **Purpose**: Model interpretability
- **Usage**:
  - Explain model predictions
  - Feature importance analysis
  - Understand model decisions
- **Implementation**:
  ```python
  explainer = shap.TreeExplainer(model)
  shap_values = explainer.shap_values(X_test)
  shap.summary_plot(shap_values, X_test)
  ```

---

### 5. **Communication & Notifications**

#### **yagmail** (`yagmail`)
- **Purpose**: Email sending
- **Usage**:
  - Send OTP for verification
  - Password reset emails
  - Notification emails
- **Implementation**:
  ```python
  yag = yagmail.SMTP(sender_email, password)
  yag.send(to=recipient, subject='OTP', contents=otp_code)
  ```

#### **captcha** (`captcha`)
- **Purpose**: Generate CAPTCHA images
- **Usage**:
  - Bot prevention
  - Secure user registration
  - Verify human users

---

### 6. **Translation & Localization**

#### **deep-translator** (`deep_translator`)
- **Purpose**: Multi-language translation
- **Usage**:
  - Translate UI text to regional languages
  - Support Hindi, Gujarati, Marathi, etc.
  - Make app accessible to farmers
- **Implementation**:
  ```python
  from deep_translator import GoogleTranslator
  translated = GoogleTranslator(source='en', target='hi').translate(text)
  ```

---

### 7. **Supporting Libraries**

#### **altair** (`altair`)
- **Purpose**: Declarative statistical visualization
- **Usage**: Interactive charts in Streamlit

#### **pillow** (`PIL`)
- **Purpose**: Image processing
- **Usage**: Handle images, logos, CAPTCHA generation

#### **requests** (`requests`)
- **Purpose**: HTTP requests
- **Usage**: API calls, external data fetching

#### **protobuf** (`protobuf`)
- **Purpose**: Data serialization
- **Usage**: Efficient data storage and transfer

---

## 🔄 Complete Project Workflow

### **Phase 1: User Registration & Authentication**

```
1. User visits application
   ↓
2. Chooses Sign Up or Login
   ↓
3. Sign Up Flow:
   - Enter email, password, username, phone
   - Password hashed with bcrypt
   - User data saved to PostgreSQL (users table)
   - JWT token generated
   ↓
4. Login Flow:
   - Enter email/phone and password
   - bcrypt verifies password hash
   - JWT token created and stored in cookies
   - Session state updated
   ↓
5. User authenticated and redirected to dashboard
```

**Libraries Used**: `streamlit`, `bcrypt`, `PyJWT`, `psycopg2`, `streamlit-cookies-manager`

---

### **Phase 2: Soil Data Entry**

```
1. User clicks "Add Soil Details"
   ↓
2. Form displays with fields:
   - State (dropdown)
   - District (dropdown - filtered by state)
   - Taluka (dropdown - filtered by district)
   - Soil Type (dropdown: Black Cotton, Loamy, Sandy, etc.)
   - pH Level (number input: 4.0-9.5)
   ↓
3. User submits form
   ↓
4. Data validated:
   - pH range check (4.0-9.5)
   - Required fields check
   ↓
5. Data saved to PostgreSQL (soil_details table)
   ↓
6. Success message displayed
   ↓
7. User redirected to dashboard
```

**Libraries Used**: `streamlit`, `psycopg2`, `pandas` (for taluka data)

---

### **Phase 3: ML Model Training (Backend - One-time)**

```
1. Load dataset (gujarat_full_crop_dataset.csv)
   ↓
2. Data Cleaning:
   - Remove duplicates
   - Handle missing values
   - Remove outliers
   - Correct mislabeled records
   ↓
3. Feature Engineering:
   - Create NPK ratios
   - Cyclical month encoding (sin/cos)
   - Temperature interactions
   - pH optimality features
   ↓
4. Data Preprocessing:
   - Label encoding (crops, districts, talukas)
   - One-hot encoding (soil types)
   - Feature scaling (StandardScaler)
   ↓
5. Class Balancing:
   - Apply SMOTE to balance crop classes
   - Calculate class weights
   ↓
6. Model Training:
   a) XGBoost Classifier (Crop Recommendation)
      - 400 estimators, max_depth=6
      - L1/L2 regularization
      - 5-fold cross-validation
   
   b) Random Forest (Suitability Prediction)
      - 500 estimators, max_depth=15
      - Calibrated probabilities
   
   c) XGBoost Regressor (Yield Prediction)
      - 500 estimators, max_depth=8
      - Crop-specific yield ranges
   ↓
7. Model Evaluation:
   - Accuracy, F1-score, Precision, Recall
   - Confusion matrices
   - ROC curves
   - Feature importance analysis
   ↓
8. Save Models:
   - crop_recommendation_models.pkl (all models)
   - crop_treatments.json (treatment database)
```

**Libraries Used**: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `matplotlib`, `seaborn`, `shap`

---

### **Phase 4: Crop Prediction (Normal Mode)**

```
1. User clicks "Crop Prediction"
   ↓
2. System retrieves user's soil details from database
   ↓
3. Prepare input features:
   - Load soil data (type, pH, district, taluka)
   - Add default weather parameters
   - Engineer features (NPK ratios, cyclical encoding)
   - Scale features using StandardScaler
   ↓
4. Load trained models from .pkl file
   ↓
5. Run predictions:
   a) XGBoost predicts top crops with probabilities
   b) Random Forest checks suitability for each crop
   c) Apply domain knowledge rules (e.g., Tobacco-Anand fix)
   d) XGBoost predicts yield for suitable crops
   ↓
6. Post-processing:
   - Filter crops with suitability > 35%
   - Rank by combined probability and suitability
   - Apply realistic yield ranges
   - Select top 3-4 crops
   ↓
7. Retrieve treatment plans from crop_treatments.json
   ↓
8. Save prediction to database (prediction_history table)
   ↓
9. Display results:
   - Top recommended crops with probabilities
   - Suitability confidence scores
   - Predicted yields
   - Treatment plans (fertilizers, pesticides, irrigation)
   ↓
10. Optional: Translate results to regional language
```

**Libraries Used**: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `psycopg2`, `streamlit`, `deep-translator`

---

### **Phase 5: Crop Prediction (Advanced Mode)**

```
1. User selects specific crop from dropdown
   ↓
2. System retrieves user's soil details
   ↓
3. Prepare input features (same as normal mode)
   ↓
4. Run advanced prediction:
   a) Random Forest predicts suitability (Yes/No)
   b) Apply domain knowledge rules for selected crop
   c) Calculate confidence score
   d) XGBoost predicts expected yield
   ↓
5. Generate detailed analysis:
   - Suitability decision with reasoning
   - Confidence percentage
   - Predicted yield with realistic range
   - Detailed treatment plan
   - Soil-crop compatibility explanation
   ↓
6. Save prediction to database
   ↓
7. Display comprehensive results:
   - "Grow" or "Not Grow" recommendation
   - Confidence score
   - Reasoning (e.g., "IDEAL: Sandy Loam soil with pH 6.5")
   - Predicted yield
   - Complete treatment plan
   ↓
8. Optional: Translate to regional language
```

**Libraries Used**: Same as Normal Mode

---

### **Phase 6: Results Translation**

```
1. User selects language (Hindi, Gujarati, etc.)
   ↓
2. System extracts text fields to translate:
   - Crop names
   - Treatment recommendations
   - Reasoning text
   - Notes and instructions
   ↓
3. GoogleTranslator API called for each field
   ↓
4. Translated text displayed alongside original
   ↓
5. User can compare original and translated versions
```

**Libraries Used**: `deep-translator`, `streamlit`

---

### **Phase 7: Prediction History**

```
1. User clicks "History" button
   ↓
2. System queries prediction_history table
   ↓
3. Retrieve all user predictions:
   - Prediction type (normal/advanced)
   - Predicted crops
   - Confidence scores
   - Timestamps
   - Full prediction results (JSON)
   ↓
4. Display in organized format:
   - Recent predictions (last 3)
   - Statistics (total predictions, most predicted crop)
   - Detailed view with expandable JSON
   ↓
5. User can view full details of any prediction
```

**Libraries Used**: `psycopg2`, `streamlit`, `pandas`

---

## 📊 Data Flow Diagram

```
┌─────────────┐
│   User      │
└──────┬──────┘
       │
       ↓
┌─────────────────────────────────────┐
│   Streamlit Web Interface           │
│   (main.py, page/*.py)              │
└──────┬──────────────────────────────┘
       │
       ├──→ Authentication (bcrypt, JWT)
       │
       ├──→ Database Operations (psycopg2)
       │    └──→ PostgreSQL Database
       │         ├── users table
       │         ├── soil_details table
       │         └── prediction_history table
       │
       ├──→ ML Prediction (utils/model_integration.py)
       │    │
       │    ├──→ Load Models (.pkl file)
       │    │    ├── XGBoost Classifier
       │    │    ├── Random Forest
       │    │    └── XGBoost Regressor
       │    │
       │    ├──→ Feature Engineering (pandas, numpy)
       │    │
       │    ├──→ Domain Knowledge Rules
       │    │
       │    └──→ Prediction Results
       │
       ├──→ Translation (deep-translator)
       │
       └──→ Results Display (streamlit)
```

---

## 🎯 Key Integration Points

### **1. Database ↔ Application**
- **Connection**: `psycopg2.connect(**db_params)`
- **Operations**: CRUD operations for users, soil details, predictions
- **Security**: Parameterized queries, password hashing

### **2. ML Models ↔ Application**
- **Loading**: `pickle.load(open('crop_recommendation_models.pkl', 'rb'))`
- **Integration**: `utils/model_integration.py` - CropPredictionService class
- **Prediction**: Real-time inference on user input

### **3. Frontend ↔ Backend**
- **State Management**: `st.session_state` for user data
- **Navigation**: Page routing through session state
- **Data Flow**: Form inputs → Processing → Database → Display

### **4. Translation ↔ Results**
- **Trigger**: User selects language
- **Process**: Extract text → Translate → Display
- **Libraries**: `deep-translator` with Google Translate API

---

## 🚀 Deployment Architecture

```
┌──────────────────────────────────────┐
│   Railway/Cloud Platform             │
│                                      │
│   ┌────────────────────────────┐   │
│   │   Docker Container         │   │
│   │                            │   │
│   │   ┌──────────────────┐    │   │
│   │   │  Streamlit App   │    │   │
│   │   │  (Port 8080)     │    │   │
│   │   └────────┬─────────┘    │   │
│   │            │               │   │
│   │            ↓               │   │
│   │   ┌──────────────────┐    │   │
│   │   │  ML Models       │    │   │
│   │   │  (.pkl files)    │    │   │
│   │   └──────────────────┘    │   │
│   └────────────────────────────┘   │
│            │                        │
│            ↓                        │
│   ┌────────────────────────────┐   │
│   │  PostgreSQL Database       │   │
│   │  (Railway Postgres)        │   │
│   └────────────────────────────┘   │
└──────────────────────────────────────┘
```

---

## 📝 Summary

This Gujarat Crop Recommendation System integrates:
- **15+ Python libraries** for web, ML, database, and translation
- **3 ML models** (XGBoost Classifier, Random Forest, XGBoost Regressor)
- **PostgreSQL database** with 3 main tables
- **Multi-language support** for 12+ Indian languages
- **Real-time predictions** with domain knowledge integration
- **Secure authentication** with bcrypt and JWT
- **Comprehensive workflow** from registration to prediction to history

The system successfully combines modern web development, machine learning, and agricultural domain expertise to provide accurate, accessible crop recommendations for Gujarat farmers.