use omega_lang::stdlib::web::router::{Router, HttpMethod, UrlBuilder};
use omega_lang::stdlib::web::middleware::{Request, Response};
use omega_lang::stdlib::web::template::Html;
use omega_lang::stdlib::web::session::{Session, SessionStore};

#[test]
fn test_route_matching() {
    let router = Router::new()
        .get("/", "home")
        .get("/users/:id", "user_detail")
        .post("/users", "create_user");

    let matched = router.match_route(&HttpMethod::GET, "/users/42").unwrap();
    assert_eq!(matched.handler, "user_detail");
    assert_eq!(matched.params.get("id").unwrap(), "42");
}

#[test]
fn test_no_match() {
    let router = Router::new().get("/", "home");
    assert!(router.match_route(&HttpMethod::POST, "/").is_none());
}

#[test]
fn test_route_group() {
    let api_routes = Router::new()
        .get("/users", "list_users")
        .get("/users/:id", "get_user");

    let router = Router::new()
        .group("/api", api_routes);

    let matched = router.match_route(&HttpMethod::GET, "/api/users/5").unwrap();
    assert_eq!(matched.handler, "get_user");
}

#[test]
fn test_url_builder() {
    let url = UrlBuilder::new("https", "example.com")
        .port(8080)
        .path("/api/users")
        .query("page", "1")
        .build();
    assert_eq!(url, "https://example.com:8080/api/users?page=1");
}

#[test]
fn test_request() {
    let mut req = Request::new("GET", "/api/users");
    req.headers.insert("content-type".to_string(), "application/json".to_string());
    assert!(req.is_json());
    assert_eq!(req.header("content-type"), Some("application/json"));
}

#[test]
fn test_response_builder() {
    let res = Response::ok()
        .with_header("X-Custom", "value")
        .with_body("hello");
    assert_eq!(res.status, 200);
    assert_eq!(res.body, "hello");
    assert_eq!(res.header("X-Custom"), Some("value"));
}

#[test]
fn test_json_response() {
    let res = Response::json(200, r#"{"status":"ok"}"#);
    assert_eq!(res.header("Content-Type"), Some("application/json"));
}

#[test]
fn test_redirect() {
    let res = Response::redirect("/login");
    assert_eq!(res.status, 302);
    assert_eq!(res.header("Location"), Some("/login"));
}

#[test]
fn test_html_escape() {
    assert_eq!(Html::escape("<script>"), "&lt;script&gt;");
    assert_eq!(Html::escape("a&b"), "a&amp;b");
}

#[test]
fn test_html_table() {
    let headers = vec!["Name", "Age"];
    let rows = vec![vec!["Alice", "30"]];
    let html = Html::table(&headers, &rows);
    assert!(html.contains("<th>Name</th>"));
    assert!(html.contains("<td>Alice</td>"));
}

#[test]
fn test_session() {
    let mut session = Session::new();
    session.set("user_id", "42");
    assert_eq!(session.get("user_id"), Some("42"));
    session.remove("user_id");
    assert!(session.get("user_id").is_none());
}

#[test]
fn test_session_store() {
    let mut store = SessionStore::new();
    store.create();
    assert_eq!(store.count(), 1);
}

#[test]
fn test_from_cookie() {
    let cookie = "session_id=abc123; Path=/";
    assert_eq!(SessionStore::from_cookie(cookie), Some("abc123"));
}
