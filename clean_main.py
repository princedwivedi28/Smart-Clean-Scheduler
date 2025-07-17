import streamlit as st
import pickle
from datetime import datetime
import joblib
import pandas as pd
import sklearn

model = joblib.load('model.pkl')

st.title('Bathroom Cleaning Pridictor')

st.markdown('''This app predicts bathroom occupancy for your selected date range.
If total occupancy is over 300 people, it will advise cleaning.
It also suggests the best hour to clean — when the bathroom is least busy.''')

st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.rawpixel.com/image_800/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDI0LTA4L2hpcHBvdW5pY29ybl9saWdodF95ZWxsb3dfY29sb3JfcGFwZXJfdGV4dHVyZV9iYWNrZ3JvdW5kX21pbmltYV8xOTU1ZjVmMC0yZTYzLTQ2YjktOTlhZi1jNGViYThlZjQ4ZDVfMS5qcGc.jpg");
        background-size: cover;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

start_date = st.date_input('select start date',value=pd.Timestamp.today())
end_date = st.date_input(" select End Date",value=pd.Timestamp.today())

if st.button('Predict'):
    if start_date>end_date:
        print('❌ End date must be after start date.')
    elif pd.Timestamp(start_date)<pd.Timestamp.today().normalize():
        print('❌ Start date is in the past. Please select a future date.')
    else:
        future_dates= pd.date_range(start=start_date,end=end_date,freq='H')
        future_df = pd.DataFrame({'event_time': future_dates})
        future_df['hour'] = future_df['event_time'].dt.hour
        future_df['day_of_week'] = future_df['event_time'].dt.dayofweek
        future_df['day'] = future_df['event_time'].dt.day
        future_df['month'] = future_df['event_time'].dt.month

        x_future = future_df[['hour','day_of_week','day','month']]

        future_df['pridicted_occupancy'] = model.predict(x_future)

        st.write('### 🕒Pridicted occupacy:')
        st.dataframe(future_df)

        total_occupancy = future_df['pridicted_occupancy'].sum()

        if total_occupancy>300:
            st.warning(f'🚨 Total predicted occupancy: **{total_occupancy:.0f}** → **Clean the bathroom!**')

        else:
            st.success(f"✅ Total predicted occupancy: **{total_occupancy:.0f}** → **Cleaning not needed.**")
        
        best_time = future_df.loc[future_df['pridicted_occupancy'].idxmin(),'event_time']

        st.info(f'Suggested Time To Clean: {best_time.strftime('%Y-%m-%d %H:%M')}**(Least busy hour)')
        st.line_chart(future_df.set_index('event_time')['pridicted_occupancy'])
        
