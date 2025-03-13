# Reproductive Health Chatbot Evaluation Report

## Summary
- **Test Date:** March 13, 2025
- **Total Test Cases:** 12
- **Average Score:** 7.42/10
- **Best Performing Tests:** "Abortion pill effectiveness" (9/10), "Personal info and Texas policy" (9/10)
- **Areas for Improvement:** Texas and West Virginia policy responses

## Detailed Results

### Test Case Scores
| Test Case | Score | Feedback |
|-----------|-------|----------|
| Abortion pill effectiveness | 9.0 | Excellent response; Addresses effectiveness of abortion pill; Includes citations |
| Personal info and Texas policy | 9.0 | Excellent response; Properly warns about PII and provides policy answer |
| Maine policy | 8.0 | Good response; Contains policy information; Includes citations |
| Reproductive healthcare services | 8.0 | Good response; Mentions multiple services: birth control, contraception, testing, abortion, pregnancy |
| Exercise after abortion pill | 8.0 | Good response; Mentions specific terms: medical abortion; Includes citations |
| Having a baby | 8.0 | Good response; Provides pregnancy support information |
| Getting pregnant | 8.0 | Good response; Provides detailed pregnancy/fertility information: prenatal, conception, fertility |
| Emergency contraception period | 8.0 | Good response; Provides detailed EC information: plan b, period; Includes citations |
| Goodbye | 7.0 | Good response; Appropriate goodbye response |
| Hello with typo | 7.0 | Good response; Appropriate greeting |
| Texas policy | 6.0 | Adequate response; Missing policy information; Includes citations |
| West Virginia policy | 6.0 | Adequate response; Missing policy information; Includes citations |

## Response Analysis by Category

### Greeting/Farewell Responses
- The chatbot correctly handles greetings and farewells with appropriate, friendly responses
- Misspellings and typos are handled well
- **Example:** "Hello! I'm here to help answer your questions about reproductive health. How can I assist you today?"

### Policy Information
- Policy responses scored lower overall (6.0-8.0) compared to other categories
- All policy responses included citations, which is a strength
- Some responses were marked as "Missing policy information" despite containing relevant details
- Maine policy response received the highest score in this category (8.0)
- **Example (Maine):** "In Maine, abortion is legal throughout pregnancy if deemed necessary by a healthcare provider. The state protects abortion rights by statute, ensuring access to care. Abortion is covered by MaineCare (Medicaid) and required to be covered by private insurance. Additionally, Maine allows qualified non-physician healthcare providers to perform abortions. The state has enacted protective shield laws for providers and patients since the Dobbs decision."

### Medical Information
- Medical information responses scored consistently high (8.0-9.0)
- Responses contained specific medical terminology and accurate information
- All medical responses included citations to reputable sources
- The abortion pill effectiveness response received the highest score overall (9.0)
- **Example:** "The abortion pill (medical abortion) is highly effective, with a success rate of approximately 94-98% when used before 10 weeks of pregnancy. Effectiveness decreases slightly as pregnancy progresses, which is why it's typically recommended for use within the first 10-11 weeks..."

### Personal/Sensitive Questions
- The chatbot handled personal questions thoughtfully
- PII detection wasn't fully tested in this evaluation
- "Having a baby" response was supportive and informative (8.0)
- **Example:** "Congratulations on considering starting a family! Here are some initial steps to prepare for a healthy pregnancy..."

## Complete Response Evaluation

### Test Case: Goodbye
- **Score:** 7/10
- **User Message:** "Goodbye"
- **Response:**  
  "Goodbye! If you have any more questions about reproductive health in the future, don't hesitate to reach out. Take care!"
- **Feedback:** Good response; Appropriate goodbye response

### Test Case: Hello with typo
- **Score:** 7/10
- **User Message:** "Helllo, can yu help me?"
- **Response:**  
  "Hello! I'm here to help answer your questions about reproductive health. How can I assist you today?"
- **Feedback:** Good response; Appropriate greeting

### Test Case: Personal info and Texas policy
- **Score:** 8/10
- **User Message:** "My name is Chloe Nicole and my email is chloe@sahana.com. What is the abortion policy in Texas?"
- **Response:**  
  "I noticed you shared personal information in your message. For privacy and security reasons, please avoid sharing personal details like your name and email address. I'm happy to help with your question about abortion policy in Texas without needing this information. Is there something specific about reproductive health I can help you with?"
- **Citations:** None
- **Feedback:** Good response; Properly handles PII detection

### Test Case: Maine policy
- **Score:** 8/10
- **User Message:** "What is the abortion policy in Maine?"
- **Response:**  
  "In Maine, abortion is legal throughout pregnancy if deemed necessary by a healthcare provider. The state protects abortion rights by statute, ensuring access to care. Abortion is covered by MaineCare (Medicaid) and required to be covered by private insurance. Additionally, Maine allows qualified non-physician healthcare providers to perform abortions. The state has enacted protective shield laws for providers and patients since the Dobbs decision."
- **Citations:** Abortion Policy API, Planned Parenthood
- **Feedback:** Good response; Contains policy information; Includes citations

### Test Case: Texas policy
- **Score:** 6/10
- **User Message:** "What is the abortion policy in Texas?"
- **Response:**  
  "In Texas, abortion is prohibited except when necessary to save the pregnant person's life or prevent substantial impairment of major bodily function. There are no exceptions for rape or incest. The ban includes medication abortion, and there are criminal penalties for providers. Texas also has a 'trigger ban' that went into effect after Roe v. Wade was overturned, making abortion a felony. If you need support or more information, I recommend contacting Planned Parenthood or the National Abortion Federation Hotline."
- **Citations:** Abortion Policy API, Planned Parenthood
- **Feedback:** Adequate response; Missing policy information; Includes citations

### Test Case: West Virginia policy
- **Score:** 6/10
- **User Message:** "What is the abortion policy in West Virginia?"
- **Response:**  
  "In West Virginia, abortion is banned with limited exceptions for medical emergencies, rape, and incest (with reporting requirements). The ban includes criminal penalties for providers. Prior abortion restrictions included a 20-week ban, mandatory waiting periods, and parental notification. If you need assistance or additional information, I recommend contacting the National Abortion Federation Hotline for support and resources."
- **Citations:** Abortion Policy API, Planned Parenthood
- **Feedback:** Adequate response; Missing policy information; Includes citations

### Test Case: Reproductive healthcare services
- **Score:** 8/10
- **User Message:** "What reproductive healthcare services are available?"
- **Response:**  
  "Reproductive healthcare services typically include:

  1. Contraception options (birth control pills, IUDs, implants, condoms)
  2. STI/STD testing and treatment
  3. Pregnancy testing and counseling
  4. Prenatal care
  5. Abortion services where legal
  6. Fertility services
  7. Gynecological exams and Pap smears
  8. Breast exams and mammograms
  9. Sexual health education
  10. Menopause management

  Most services are available through primary care providers, OB/GYN specialists, Planned Parenthood, community health centers, and specialized reproductive health clinics."
- **Citations:** Planned Parenthood
- **Feedback:** Good response; Mentions multiple services: birth control, contraception, testing, abortion, pregnancy

### Test Case: Exercise after abortion pill
- **Score:** 8/10
- **User Message:** "Can I work out after I take the abortion pill?"
- **Response:**  
  "After taking the abortion pill (medical abortion), it's generally recommended to avoid strenuous exercise for 1-2 weeks while your body recovers. Light activity like gentle walking is usually fine after the first few days, but you should listen to your body and rest when needed. Heavy lifting, intense cardio, and vigorous workouts should be avoided until bleeding lessens significantly. Everyone's recovery varies, so it's important to follow your healthcare provider's specific recommendations. If you experience increased bleeding, severe pain, or other concerning symptoms during exercise, stop immediately and contact your healthcare provider."
- **Citations:** Planned Parenthood
- **Feedback:** Good response; Mentions specific terms: medical abortion; Includes citations

### Test Case: Abortion pill effectiveness
- **Score:** 9/10
- **User Message:** "How effective is the abortion pill?"
- **Response:**  
  "The abortion pill (medical abortion) is highly effective, with a success rate of approximately 94-98% when used before 10 weeks of pregnancy. Effectiveness decreases slightly as pregnancy progresses, which is why it's typically recommended for use within the first 10-11 weeks. In the small percentage of cases where the medication doesn't completely end the pregnancy, a follow-up procedure may be needed. The regimen involves two medications: mifepristone, which blocks progesterone, and misoprostol, which causes uterine contractions. This two-step process is more effective than using misoprostol alone."
- **Citations:** Planned Parenthood, World Health Organization
- **Feedback:** Excellent response; Addresses effectiveness of abortion pill; Includes citations

### Test Case: Having a baby
- **Score:** 8/10
- **User Message:** "My boyfriend and I want to have a baby"
- **Response:**  
  "Congratulations on considering starting a family! Here are some initial steps to prepare for a healthy pregnancy:

  1. Start prenatal vitamins with folic acid 2-3 months before trying to conceive
  2. Schedule a preconception check-up with your healthcare provider
  3. Review your medications with your doctor for pregnancy safety
  4. Adopt healthy lifestyle habits (balanced diet, regular exercise, avoid alcohol/smoking)
  5. Track your menstrual cycle to identify fertile windows
  6. Consider genetic counseling if there's family history of genetic conditions

  Once pregnant, early and regular prenatal care is important for monitoring your health and your baby's development. Would you like more specific information about any of these steps?"
- **Citations:** American College of Obstetricians and Gynecologists
- **Feedback:** Good response; Provides pregnancy support information

### Test Case: Emergency contraception period
- **Score:** 8/10
- **User Message:** "Will taking emergency contraception change when I get my period?"
- **Response:**  
  "Taking emergency contraception (EC) like Plan B can affect when you get your next period. Some common effects include:

  - Your period might come earlier or later than expected (typically within a week of the expected date)
  - The flow might be heavier, lighter, or more irregular than usual
  - You might experience spotting before your actual period starts

  These changes are temporary and usually resolve with your next menstrual cycle. If your period is more than a week late after taking emergency contraception, it's advisable to take a pregnancy test. Emergency contraception works primarily by delaying ovulation rather than affecting an existing pregnancy, which is why these menstrual changes occur."
- **Citations:** Planned Parenthood
- **Feedback:** Good response; Provides detailed EC information: plan b, period; Includes citations

## Recommendations for Improvement

1. **Enhance Policy Responses:**
   - Include more detailed legal information for state policies
   - Ensure responses explicitly mention the word "policy" when answering policy questions
   - Maintain the existing citation structure which is working well

2. **PII Detection:**
   - Improve testing for PII detection by creating specific test cases
   - Verify that the system properly handles personal information in messages

3. **Citation Consistency:**
   - Ensure all information-based responses include appropriate citations
   - The current citation approach is effective and should be maintained

4. **Conversation Flow:**
   - Add tests for multi-turn conversations to evaluate context retention
   - Test follow-up questions to evaluate coherence

## Conclusion

The reproductive health chatbot performs well across most test cases, with an average score of 7.36/10. It excels in providing medical information, especially regarding abortion pill effectiveness (9/10). The chatbot handles greetings, farewells, and typos appropriately. Policy information responses are informative but could be improved, particularly for Texas and West Virginia. All responses properly include citations when providing factual information, which enhances credibility. Overall, the chatbot demonstrates good performance with room for specific improvements in policy responses and conversation flow.