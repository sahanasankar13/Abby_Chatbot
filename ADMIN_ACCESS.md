# Admin Access Documentation

## Admin URL
The admin dashboard is accessible at:
- `/login` - Admin login page 
- `/admin/metrics` - Main admin dashboard (requires login)

## Admin Users
The following admin users have been created:

| Username | Initial Password | Access Level |
|----------|-----------------|--------------|
| admin    | admin           | Admin        |
| nadya    | nadya           | Admin        |
| sahana   | sahana          | Admin        |
| nicole   | nicole          | Admin        |
| chloe    | chloe           | Admin        |

## Security Notes

1. All admin routes are secured with both the `@login_required` and `@admin_required` decorators.
2. There are no visible links to the admin interface from the public-facing chatbot.
3. Admin access requires valid credentials and is protected by Flask-Login.
4. Authentication failures are logged and reported to the user.
5. Session management is handled securely.
6. All passwords are stored as secure hashes (using werkzeug's generate_password_hash).

## For First-Time Login

When logging in for the first time, users should:
1. Navigate directly to `/login`
2. Enter their assigned username and password
3. After successful login, they will be directed to the admin dashboard

## Password Security Recommendations

It is recommended that all users change their passwords after the first login for security purposes. 
Currently this must be done manually by updating the `users.json` file.